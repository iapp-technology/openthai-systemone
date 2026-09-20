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
    @torch.no_grad()
    def system_one(
        self,
        state: Union[str, Dict[str, Any], List[Any]],
        questions: Dict[str, Union[Question, Dict[str, Any]]],
    ) -> SystemOneResponse:
        return self.system_one_batch([(state, questions)])[0]

    @torch.no_grad()
    def system_one_batch(
        self,
        items: Sequence[tuple],
    ) -> List[SystemOneResponse]:
        parsed = [(s, {k: parse_question(v) for k, v in qs.items()}) for s, qs in items]
        encs = [self.fmt.encode(s, qs) for s, qs in parsed]
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
        probs = out.probs.float().cpu()
        return [self._decode(e, probs[b]) for b, e in enumerate(encs)]

    # ------------------------------------------------------------------ decoding
    def _decode(self, enc: Encoded, probs: torch.Tensor) -> SystemOneResponse:
        answers: Dict[str, Answer] = {}
        for qi, spec in enumerate(enc.specs):
            p = probs[qi]
            k = len(spec.option_names)
            abstain = float(p[self.model.config.abstain_slot])
            pk = p[:k]
            pk = pk / pk.sum().clamp(min=1e-12)  # renormalise over the real options
            conf = confidence_from_probs(pk, k)
            if spec.qtype == "noul":
                answers[spec.qid] = NoulAnswer(noul=float(pk[1]))
            elif spec.qtype == "choice":
                prob_map = {spec.option_names[i]: float(pk[i]) for i in range(k)}
                best = max(prob_map, key=prob_map.get)
                answers[spec.qid] = ChoiceAnswer(choice=best, probabilities=prob_map, confidence=conf, abstain=abstain)
            else:  # score
                levels = torch.arange(k, dtype=torch.float32)
                score = float((pk * levels).sum())
                answers[spec.qid] = ScoreAnswer(
                    score=score,
                    legend={i: spec.option_descs[i] or str(i) for i in range(k)},
                    probabilities={str(i): float(pk[i]) for i in range(k)},
                    confidence=conf,
                )
        return SystemOneResponse(model=self.model_name, answers=answers, usage=Usage(input_tokens=enc.n_tokens))
