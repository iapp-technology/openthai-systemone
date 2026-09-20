"""OpenThai-SystemOne decision model.

text tower (Qwen3.5-0.8B, LM head removed)  ->  hidden state at every <|ts_answer|>  ->  SlotHead (256 logits)
                                                 mask slots >= k  ->  softmax  ->  probabilities over the k options
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.utils import ModelOutput

from .configuration import OpenThaiSystemOneConfig
from .formatting import SPECIAL_TOKENS, TOK_ANSWER, add_special_tokens

QTYPE_INDEX = {"choice": 0, "score": 1, "noul": 2}


@dataclass
class DecisionOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None  # (B, Q, n_slots), masked with -inf
    probs: Optional[torch.Tensor] = None  # (B, Q, n_slots)
    hidden_states: Optional[torch.Tensor] = None  # (B, Q, H) at answer positions


class OpenThaiSystemOneForDecision(PreTrainedModel):
    config_class = OpenThaiSystemOneConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, config: OpenThaiSystemOneConfig):
        super().__init__(config)
        self.model = AutoModel.from_config(config.text_config)
        self.slot_head = nn.Linear(config.hidden_size, config.n_slots, bias=config.head_bias)
        # log-temperatures per question type (choice/score/noul); learned in the calibration stage
        self.log_temperature = nn.Parameter(torch.zeros(config.n_temperatures))
        self.post_init()

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_causal_lm(
        cls,
        path: str,
        *,
        tokenizer=None,
        n_slots: int = 256,
        torch_dtype=torch.bfloat16,
        **kwargs,
    ):
        """Build a decision model from a (text-only) causal-LM checkpoint: drop lm_head, add tokens + head."""
        tok = tokenizer or AutoTokenizer.from_pretrained(path)
        added = add_special_tokens(tok)
        lm = AutoModelForCausalLM.from_pretrained(path, dtype=torch_dtype, **kwargs)
        base = lm.model if hasattr(lm, "model") else lm.base_model
        text_cfg = base.config
        if added:
            lm.resize_token_embeddings(len(tok), mean_resizing=False)
            text_cfg.vocab_size = lm.get_input_embeddings().weight.shape[0]
            _init_new_token_embeddings(lm.get_input_embeddings().weight, tok, added)
        cfg = OpenThaiSystemOneConfig(
            text_config=text_cfg,
            n_slots=n_slots,
            answer_token_id=tok.convert_tokens_to_ids(TOK_ANSWER),
            pad_token_id=tok.pad_token_id,
        )
        cfg.text_config.tie_word_embeddings = False  # there is no LM head any more
        model = cls(cfg).to(torch_dtype)
        missing, unexpected = model.model.load_state_dict(base.state_dict(), strict=False)
        assert not unexpected, unexpected
        _init_slot_head(model.slot_head)
        model.model.config = cfg.text_config
        return model, tok

    # ------------------------------------------------------------------ forward
    def gather_answer_states(self, hidden: torch.Tensor, answer_positions: torch.Tensor) -> torch.Tensor:
        idx = answer_positions.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        return torch.gather(hidden, 1, idx)  # (B, Q, H)

    def slot_logits(
        self,
        answer_hidden: torch.Tensor,
        option_counts: torch.Tensor,
        *,
        include_abstain: bool = True,
        qtypes: Optional[torch.Tensor] = None,
        apply_temperature: bool = True,
    ) -> torch.Tensor:
        logits = self.slot_head(answer_hidden.to(self.slot_head.weight.dtype)).float()
        if apply_temperature:
            if qtypes is None:
                t = self.log_temperature[0].exp()
            else:
                t = self.log_temperature.exp()[qtypes.clamp(min=0)]  # (B, Q)
                t = t.unsqueeze(-1)
            logits = logits / t
        ar = torch.arange(logits.shape[-1], device=logits.device)
        valid = ar[None, None, :] < option_counts.unsqueeze(-1)
        if include_abstain:
            valid = valid.clone()
            valid[..., self.config.abstain_slot] = True
        # questions that are padding (option_counts == 0) keep slot 0 valid to avoid NaNs
        valid[..., 0] |= option_counts.unsqueeze(-1).squeeze(-1) == 0
        return logits.masked_fill(~valid, float("-inf"))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        answer_positions: Optional[torch.Tensor] = None,
        option_counts: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        soft_labels: Optional[torch.Tensor] = None,
        qtypes: Optional[torch.Tensor] = None,
        include_abstain: bool = True,
        label_smoothing: float = 0.0,
        brier_weight: float = 0.0,
        apply_temperature: bool = True,
        **kwargs,
    ) -> DecisionOutput:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden = out.last_hidden_state
        if answer_positions is None:
            answer_positions = (input_ids == self.config.answer_token_id).nonzero()[:, 1].unsqueeze(0)
            if option_counts is None:
                raise ValueError("option_counts required")
        h = self.gather_answer_states(hidden, answer_positions)
        logits = self.slot_logits(h, option_counts, include_abstain=include_abstain, qtypes=qtypes, apply_temperature=apply_temperature)
        probs = logits.softmax(-1)

        loss = None
        if labels is not None or soft_labels is not None:
            logp = logits.log_softmax(-1)
            if soft_labels is not None:
                valid = (option_counts > 0)
                tgt = soft_labels.float()
                nll = -(tgt * logp.masked_fill(torch.isinf(logp), 0.0)).sum(-1)
                loss = (nll * valid).sum() / valid.sum().clamp(min=1)
            else:
                flat_logp = logp.reshape(-1, logp.shape[-1])
                flat_lab = labels.reshape(-1)
                keep = flat_lab != -100
                if keep.any():
                    lp = flat_logp[keep]
                    lb = flat_lab[keep]
                    nll = -lp.gather(1, lb[:, None]).squeeze(1)
                    if label_smoothing > 0:
                        n_valid = torch.isfinite(lp).sum(-1).clamp(min=1).float()
                        smooth = -(lp.masked_fill(torch.isinf(lp), 0.0)).sum(-1) / n_valid
                        nll = (1 - label_smoothing) * nll + label_smoothing * smooth
                    loss = nll.mean()
                    if brier_weight > 0:
                        p = lp.exp()
                        onehot = F.one_hot(lb, p.shape[-1]).float()
                        loss = loss + brier_weight * ((p - onehot) ** 2).sum(-1).mean()
                else:
                    loss = logits.sum() * 0.0
        return DecisionOutput(loss=loss, logits=logits, probs=probs, hidden_states=h)


def use_reference_kernels():
    """Force the pure-PyTorch Gated-DeltaNet / causal-conv paths.

    transformers routes `chunk_gated_delta_rule` & co. to the Triton kernels (flash-linear-attention, causal-conv1d)
    whenever those packages are importable, without checking the tensor device, which crashes on CPU/MPS.
    Call this before running on a non-CUDA device.
    """
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    except Exception:  # pragma: no cover
        return
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "chunk_gated_delta_rule",
                 "fused_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update"):
        fn = getattr(m, name, None)
        if fn is not None and hasattr(fn, "__wrapped__"):
            setattr(m, name, fn.__wrapped__)


def _init_slot_head(head: nn.Linear):
    nn.init.normal_(head.weight, std=0.02)
    if head.bias is not None:
        nn.init.zeros_(head.bias)


@torch.no_grad()
def _init_new_token_embeddings(weight: torch.Tensor, tok, n_added: int):
    """New control tokens start near the mean of digit-token embeddings + small noise."""
    digit_ids = [tok.convert_tokens_to_ids(d) for d in "0123456789"]
    digit_ids = [i for i in digit_ids if i is not None and i != tok.unk_token_id]
    mean = weight[digit_ids].float().mean(0) if digit_ids else weight[: weight.shape[0] - n_added].float().mean(0)
    std = weight[: weight.shape[0] - n_added].float().std()
    new = mean[None, :] + torch.randn(n_added, weight.shape[1]) * std * 0.1
    weight[-n_added:] = new.to(weight.dtype)


def confidence_from_probs(p: torch.Tensor, k: int) -> float:
    """1 - normalised entropy over the k valid options."""
    if k <= 1:
        return 1.0
    p = p[:k].clamp(min=1e-12)
    p = p / p.sum()
    h = -(p * p.log()).sum().item()
    return float(max(0.0, min(1.0, 1.0 - h / math.log(k))))
