"""Turn (state, questions) into token ids + slot bookkeeping.

Layout (one sequence, causal):

    <|ts_state|> {state text}
    <|ts_q|><|ts_choice|> {instructions}
    <|ts_opt_0|> {option name}: {description}
    <|ts_opt_1|> {option name}
    ...
    <|ts_answer|>                      <- hidden state here -> SlotHead (256 logits)
    <|ts_q|><|ts_noul|> {instructions}
    <|ts_opt_0|> no
    <|ts_opt_1|> yes
    <|ts_answer|>
    ...

Slot i (0..254) means "the option introduced by <|ts_opt_i|>"; slot 255 = abstain.
All answers for all questions are read out from one forward pass.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .types import Choice, Noul, Score, Question, MAX_OPTIONS

N_SLOTS = 256
ABSTAIN_SLOT = 255

TOK_STATE = "<|ts_state|>"
TOK_Q = "<|ts_q|>"
TOK_CHOICE = "<|ts_choice|>"
TOK_SCORE = "<|ts_score|>"
TOK_NOUL = "<|ts_noul|>"
TOK_ANSWER = "<|ts_answer|>"
TOK_OPT = [f"<|ts_opt_{i}|>" for i in range(N_SLOTS)]

SPECIAL_TOKENS: List[str] = [TOK_STATE, TOK_Q, TOK_CHOICE, TOK_SCORE, TOK_NOUL, TOK_ANSWER] + TOK_OPT

NOUL_OPTIONS = ("no", "yes")  # slot 0 = no, slot 1 = yes  -> noul = p(slot 1)

DEFAULT_MAX_TOTAL_TOKENS = 65536
DEFAULT_MAX_STATE_TOKENS = 32768


def add_special_tokens(tokenizer) -> int:
    """Register the TypeSafe control tokens. Returns number of tokens added."""
    existing = set(tokenizer.get_vocab())
    new = [t for t in SPECIAL_TOKENS if t not in existing]
    if not new:
        return 0
    return tokenizer.add_tokens(new, special_tokens=True)


def sanitize(text: str) -> str:
    """Stop user content from smuggling control tokens into the sequence."""
    return text.replace("<|ts_", "<​|ts_") if "<|ts_" in text else text


def state_to_text(state: Union[str, Dict[str, Any], List[Any]], *, indent: Optional[int] = None) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=indent)


@dataclass
class QuestionSpec:
    """A question flattened to option strings + slot bookkeeping."""

    qid: str
    qtype: str  # choice | score | noul
    instructions: str
    option_names: List[str]  # in slot order (after any permutation)
    option_descs: List[Optional[str]]
    perm: List[int]  # perm[slot] = original index of the option at that slot
    label_slot: Optional[int] = None  # training only


def question_to_spec(
    qid: str,
    q: Question,
    *,
    label: Optional[Union[str, int, bool]] = None,
    shuffle: bool = False,
    rng: Optional[random.Random] = None,
    drop_label: bool = False,
) -> QuestionSpec:
    """Flatten a typed question.

    label: for training. Choice -> option name; Score -> level index (int); Noul -> bool.
    shuffle: permute option order (Choice only; Score/Noul order is semantic).
    drop_label: remove the correct option from a Choice so the target becomes ABSTAIN_SLOT.
    """
    if isinstance(q, Choice):
        names = list(q.criteria.keys())
        descs = [q.criteria[n] for n in names]
        idx = list(range(len(names)))
        label_idx = None
        if label is not None:
            if label not in q.criteria:
                raise ValueError(f"label {label!r} is not one of the options")
            label_idx = names.index(str(label))
        if drop_label and label_idx is not None:
            if len(idx) < 2:
                raise ValueError("cannot drop the only option")
            idx.remove(label_idx)
            label_idx = None
        if shuffle:
            (rng or random).shuffle(idx)
        names_p = [names[i] for i in idx]
        descs_p = [descs[i] for i in idx]
        if label is None:
            slot = None
        elif label_idx is None:
            slot = ABSTAIN_SLOT
        else:
            slot = idx.index(label_idx)
        return QuestionSpec(qid, "choice", q.instructions, names_p, descs_p, idx, slot)

    if isinstance(q, Score):
        names = [str(i) for i in range(len(q.criteria))]
        descs = list(q.criteria)
        slot = int(label) if label is not None else None
        if slot is not None and not (0 <= slot < len(descs)):
            raise ValueError("score label out of range")
        return QuestionSpec(qid, "score", q.instructions, names, descs, list(range(len(names))), slot)

    if isinstance(q, Noul):
        c = q.criteria or {}
        descs = [c.get("false"), c.get("true")]
        slot = None if label is None else int(bool(label))
        return QuestionSpec(qid, "noul", q.instructions, list(NOUL_OPTIONS), descs, [0, 1], slot)

    raise TypeError(type(q))


def spec_to_text(spec: QuestionSpec) -> str:
    head = {"choice": TOK_CHOICE, "score": TOK_SCORE, "noul": TOK_NOUL}[spec.qtype]
    lines = [f"{TOK_Q}{head} {sanitize(spec.instructions).strip()}"]
    for i, (name, desc) in enumerate(zip(spec.option_names, spec.option_descs)):
        name = sanitize(str(name)).strip()
        if desc:
            lines.append(f"{TOK_OPT[i]} {name}: {sanitize(str(desc)).strip()}")
        else:
            lines.append(f"{TOK_OPT[i]} {name}")
    lines.append(TOK_ANSWER)
    return "\n".join(lines) + "\n"


@dataclass
class Encoded:
    input_ids: List[int]
    answer_positions: List[int]  # index of each <|ts_answer|> token, question order
    option_counts: List[int]  # k per question (valid slots 0..k-1)
    specs: List[QuestionSpec]
    labels: List[int] = field(default_factory=list)  # -100 if unknown
    truncated_state: bool = False

    @property
    def n_tokens(self) -> int:
        return len(self.input_ids)


class Formatter:
    """Tokenizer-aware encoder shared by training and inference."""

    def __init__(
        self,
        tokenizer,
        *,
        max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
        max_state_tokens: int = DEFAULT_MAX_STATE_TOKENS,
    ):
        self.tok = tokenizer
        add_special_tokens(self.tok)
        self.max_total_tokens = max_total_tokens
        self.max_state_tokens = max_state_tokens
        self.answer_id = self.tok.convert_tokens_to_ids(TOK_ANSWER)
        self.state_id = self.tok.convert_tokens_to_ids(TOK_STATE)
        self.opt_ids = self.tok.convert_tokens_to_ids(TOK_OPT)
        assert self.answer_id is not None and self.answer_id != self.tok.unk_token_id

    def _ids(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def encode(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Question],
        *,
        labels: Optional[Dict[str, Union[str, int, bool]]] = None,
        shuffle_options: bool = False,
        shuffle_questions: bool = False,
        drop_label_for: Optional[Sequence[str]] = None,
        rng: Optional[random.Random] = None,
        state_indent: Optional[int] = None,
    ) -> Encoded:
        rng = rng or random.Random()
        labels = labels or {}
        drop = set(drop_label_for or [])
        qids = list(questions.keys())
        if shuffle_questions:
            rng.shuffle(qids)

        specs = [
            question_to_spec(
                qid,
                questions[qid],
                label=labels.get(qid),
                shuffle=shuffle_options,
                rng=rng,
                drop_label=qid in drop,
            )
            for qid in qids
        ]
        q_texts = [spec_to_text(s) for s in specs]
        q_ids = [self._ids(t) for t in q_texts]
        q_total = sum(len(x) for x in q_ids)

        state_text = sanitize(state_to_text(state, indent=state_indent))
        state_ids = self._ids(TOK_STATE + " " + state_text.strip() + "\n")
        budget = min(self.max_state_tokens, self.max_total_tokens - q_total)
        truncated = False
        if len(state_ids) > budget:
            # keep the head (state token) and the tail of the state; the end is usually the most recent info
            keep_tail = max(budget - 1, 0)
            state_ids = state_ids[:1] + state_ids[len(state_ids) - keep_tail :]
            truncated = True

        ids: List[int] = list(state_ids)
        answer_positions: List[int] = []
        for qi in q_ids:
            ids.extend(qi)
            # the answer token is the last non-newline token of each question block
            pos = len(ids) - 1
            while ids[pos] != self.answer_id:
                pos -= 1
            answer_positions.append(pos)

        return Encoded(
            input_ids=ids,
            answer_positions=answer_positions,
            option_counts=[len(s.option_names) for s in specs],
            specs=specs,
            labels=[(-100 if s.label_slot is None else s.label_slot) for s in specs],
            truncated_state=truncated,
        )


def slot_mask(option_counts: Sequence[int], *, include_abstain: bool = True, n_slots: int = N_SLOTS):
    """Boolean mask (Q, n_slots): True where a slot is valid for that question."""
    import torch

    k = torch.as_tensor(list(option_counts), dtype=torch.long)
    ar = torch.arange(n_slots)
    mask = ar[None, :] < k[:, None]
    if include_abstain:
        mask[:, ABSTAIN_SLOT] = True
    return mask


def collate(encoded: Sequence[Encoded], pad_id: int, *, max_questions: Optional[int] = None):
    """Right-pad a batch. Returns dict of tensors for OpenThaiSystemOneForDecision.forward."""
    import torch

    B = len(encoded)
    T = max(e.n_tokens for e in encoded)
    Q = max_questions or max(len(e.answer_positions) for e in encoded)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.long)
    answer_positions = torch.zeros((B, Q), dtype=torch.long)
    option_counts = torch.zeros((B, Q), dtype=torch.long)
    labels = torch.full((B, Q), -100, dtype=torch.long)
    for b, e in enumerate(encoded):
        n = e.n_tokens
        input_ids[b, :n] = torch.tensor(e.input_ids)
        attention_mask[b, :n] = 1
        q = len(e.answer_positions)
        answer_positions[b, :q] = torch.tensor(e.answer_positions)
        option_counts[b, :q] = torch.tensor(e.option_counts)
        if e.labels:
            labels[b, :q] = torch.tensor(e.labels)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "answer_positions": answer_positions,
        "option_counts": option_counts,
        "labels": labels,
    }
