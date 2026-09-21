"""Local inference: `client.system_one(state, questions)` -> answers, mirroring the TypeSafe SDK."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import torch
from transformers import AutoTokenizer

from .formatting import Encoded, Formatter, collate
from .modeling import OpenThaiSystemOneForDecision, QTYPE_INDEX, confidence_from_probs, use_reference_kernels
from .types import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
    parse_question,
)


class SystemOneClient:
    def __init__(
        self,
        model_path: str,
        *,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        max_total_tokens: int = 65536,
        max_state_tokens: int = 32768,
        model_name: str = "openthai-systemone",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        if dtype is None:
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.device = device
        if device != "cuda":
            use_reference_kernels()
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = OpenThaiSystemOneForDecision.from_pretrained(model_path, dtype=dtype).to(device).eval()
        self.fmt = Formatter(self.tok, max_total_tokens=max_total_tokens, max_state_tokens=max_state_tokens)
        self.model_name = model_name
        pad = self.tok.pad_token_id
        self.pad_id = pad if pad is not None else (self.tok.eos_token_id or 0)

    # ------------------------------------------------------------------ public API
    AUTO_INVARIANT_MIN_OPTIONS = 11  # auto mode: average orders when a choice question has this many options or more
    AUTO_PERMUTATIONS = 8

    @torch.no_grad()
    def system_one(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Union[Question, Dict[str, Any]]],
        *,
        permutations: Optional[int] = None,
    ) -> SystemOneResponse:
        return self.system_one_batch([(state, questions)], permutations=permutations)[0]

    @staticmethod
    def _cyclic_orders(k: int, n: int) -> List[List[int]]:
        """n distinct cyclic shifts of range(k), evenly spread, identity first."""
        n = max(1, min(n, k))
        offsets = sorted({round(j * k / n) % k for j in range(n)})
        return [[(i + off) % k for i in range(k)] for off in offsets]

    @torch.no_grad()
    def _run(self, encs: Sequence[Encoded]) -> torch.Tensor:
        batch = collate(encs, self.pad_id)
        qtypes = torch.full_like(batch["option_counts"], -1)
        for b, e in enumerate(encs):
            for qi, spec in enumerate(e.specs):
                qtypes[b, qi] = QTYPE_INDEX[spec.qtype]
        batch = {k: v.to(self.device) for k, v in batch.items()}
        out = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            answer_positions=batch["answer_positions"],
            option_counts=batch["option_counts"],
            qtypes=qtypes.to(self.device),
            include_abstain=True,
        )
        return out.probs.float().cpu()

    @torch.no_grad()
    def system_one_batch(
        self,
        items: Sequence[tuple],
        *,
        permutations: Optional[int] = None,
    ) -> List[SystemOneResponse]:
        """permutations: None = automatic, 1 = single order, n = average over n cyclic option orders per choice question."""
        parsed = [(s, {k: parse_question(v) for k, v in qs.items()}) for s, qs in items]
        plans = []  # per item: list of option_orders dicts (one per permutation)
        for s, qs in parsed:
            choice_k = {qid: len(q.criteria) for qid, q in qs.items() if isinstance(q, Choice)}
            n = permutations
            if n is None:
                n = self.AUTO_PERMUTATIONS if any(k >= self.AUTO_INVARIANT_MIN_OPTIONS for k in choice_k.values()) else 1
            n = max(1, min(n, max(choice_k.values(), default=1)))
            orders = {qid: self._cyclic_orders(k, n) for qid, k in choice_k.items()}
            plans.append([{qid: ords[j % len(ords)] for qid, ords in orders.items()} for j in range(n)])
        encs, owner = [], []
        for i, ((s, qs), plan) in enumerate(zip(parsed, plans)):
            for oo in plan:
                encs.append(self.fmt.encode(s, qs, option_orders=oo))
                owner.append(i)
        probs = self._run(encs)
        # average per item, per question, keyed by option name (order-independent)
        results = []
        for i, (s, qs) in enumerate(parsed):
            idx = [j for j, o in enumerate(owner) if o == i]
            base = encs[idx[0]]
            per_q: Dict[str, Dict[str, float]] = {}
            per_q_abstain: Dict[str, float] = {}
            for j in idx:
                e = encs[j]
                for qi, spec in enumerate(e.specs):
                    p = probs[j][qi]
                    k = len(spec.option_names)
                    pk = p[:k] / p[:k].sum().clamp(min=1e-12)
                    d = per_q.setdefault(spec.qid, {})
                    for name, v in zip(spec.option_names, pk.tolist()):
                        d[name] = d.get(name, 0.0) + v / len(idx)
                    per_q_abstain[spec.qid] = per_q_abstain.get(spec.qid, 0.0) + float(p[self.model.config.abstain_slot]) / len(idx)
            results.append(self._decode_named(base, per_q, per_q_abstain, len(idx)))
        return results

    # ------------------------------------------------------------------ decoding
    def _decode_named(self, enc: Encoded, per_q: Dict[str, Dict[str, float]], abstain: Dict[str, float], n_perm: int) -> SystemOneResponse:
        answers: Dict[str, Answer] = {}
        for spec in enc.specs:
            d = per_q[spec.qid]
            k = len(spec.option_names)
            # canonical option order = the order the caller gave (identity permutation = enc built from plan[0])
            names = spec.option_names if spec.qtype != "choice" else list(d.keys())
            pk = torch.tensor([d[n] for n in names], dtype=torch.float32)
            pk = pk / pk.sum().clamp(min=1e-12)
            conf = confidence_from_probs(pk, k)
            if spec.qtype == "noul":
                answers[spec.qid] = NoulAnswer(noul=float(d["yes"]))
            elif spec.qtype == "choice":
                prob_map = {n: float(v) for n, v in zip(names, pk.tolist())}
                best = max(prob_map, key=prob_map.get)
                answers[spec.qid] = ChoiceAnswer(choice=best, probabilities=prob_map, confidence=conf, abstain=abstain.get(spec.qid))
            else:  # score: names are "0".."k-1" in level order
                levels = torch.arange(k, dtype=torch.float32)
                score = float((pk * levels).sum())
                answers[spec.qid] = ScoreAnswer(
                    score=score,
                    legend={i: spec.option_descs[i] or str(i) for i in range(k)},
                    probabilities={str(i): float(pk[i]) for i in range(k)},
                    confidence=conf,
                )
        return SystemOneResponse(model=self.model_name, answers=answers, usage=Usage(input_tokens=enc.n_tokens, permutations=n_perm))
