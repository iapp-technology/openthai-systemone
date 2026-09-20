"""Small single-GPU training helpers shared by the CPT and decision-SFT loops."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


@dataclass
class TrainConfig:
    output_dir: str
    model_path: str
    data: Dict[str, Any] = field(default_factory=dict)
    seq_len: int = 4096
    micro_batch: int = 4
    grad_accum: int = 64
    lr: float = 5e-5
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    warmup_ratio: float = 0.01
    max_steps: int = 0  # 0 -> derive from tokens/epochs
    total_tokens: float = 0  # CPT budget
    epochs: float = 1.0  # SFT
    grad_clip: float = 1.0
    bf16: bool = True
    gradient_checkpointing: bool = True
    compile: bool = False
    save_every: int = 500
    eval_every: int = 250
    log_every: int = 10
    seed: int = 0
    resume: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # PyYAML parses "5.0e9" / "1e-5" (no signed exponent) as strings; coerce every numeric field
        for f in ("lr", "min_lr_ratio", "weight_decay", "warmup_ratio", "total_tokens", "epochs", "grad_clip"):
            setattr(self, f, float(getattr(self, f)))
        for f in ("seq_len", "micro_batch", "grad_accum", "max_steps", "save_every", "eval_every", "log_every", "seed"):
            setattr(self, f, int(getattr(self, f)))
        self.betas = tuple(float(b) for b in self.betas)

    @staticmethod
    def load(path: str, **overrides) -> "TrainConfig":
        d = yaml.safe_load(open(path))
        d.update({k: v for k, v in overrides.items() if v is not None})
        return TrainConfig(**d)

    def dump(self, path: Path):
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True))


def lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    warm = max(1, int(total * cfg.warmup_ratio))
    if step < warm:
        return cfg.lr * step / warm
    prog = min(1.0, (step - warm) / max(1, total - warm))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def make_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in n.lower() or n.endswith(".bias") or "temperature" in n else decay).append(p)
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=tuple(cfg.betas), fused=fused,
    )


class Logger:
    def __init__(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        self.f = open(out / "train_log.jsonl", "a")
        self.t0 = time.time()

    def log(self, **kw):
        kw["elapsed_s"] = round(time.time() - self.t0, 1)
        self.f.write(json.dumps(kw) + "\n")
        self.f.flush()
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in kw.items()), flush=True)


def save_checkpoint(out: Path, model, tok, optimizer, step: int, extra: Optional[dict] = None):
    d = out / f"step-{step}"
    d.mkdir(parents=True, exist_ok=True)
    (model._orig_mod if hasattr(model, "_orig_mod") else model).save_pretrained(d, safe_serialization=True)
    if tok is not None:
        tok.save_pretrained(d)
    torch.save({"optimizer": optimizer.state_dict(), "step": step, **(extra or {})}, d / "trainer_state.pt")
    latest = out / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(d.name)
    return d


def device_and_dtype(cfg: TrainConfig):
    if torch.cuda.is_available():
        return torch.device("cuda"), (torch.bfloat16 if cfg.bf16 else torch.float32)
    from .modeling import use_reference_kernels

    use_reference_kernels()
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def set_seed(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
