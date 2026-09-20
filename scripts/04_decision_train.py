"""Stage 2/3: decision-head SFT (and calibration fine-tune) on unified records.

    python scripts/04_decision_train.py --config configs/sft.yaml
    python scripts/04_decision_train.py --config configs/calib.yaml --init runs/sft/latest   # Brier + temperature stage

Reads data/decision/*.train.jsonl + data/synth/*.jsonl, samples sources by weight, applies augmentation
(option shuffling, question shuffling, 3% abstain, state as JSON with random indent), packs by length,
and trains OpenThaiSystemOneForDecision with masked 256-slot cross-entropy.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openthai_systemone.data.records import Record, read_jsonl  # noqa: E402
from openthai_systemone.formatting import Formatter, collate  # noqa: E402
from openthai_systemone.modeling import OpenThaiSystemOneForDecision, QTYPE_INDEX  # noqa: E402
from openthai_systemone.train_utils import (  # noqa: E402
    Logger, TrainConfig, device_and_dtype, lr_at, make_optimizer, save_checkpoint, set_seed,
)
from openthai_systemone.types import parse_question  # noqa: E402


def load_records(globs: List[str], weights: Dict[str, float], limit_per_source: int = 0) -> Dict[str, List[Record]]:
    by_src: Dict[str, List[Record]] = defaultdict(list)
    for g in globs:
        for f in sorted(glob.glob(g)):
            for r in read_jsonl(f):
                if r.split != "train":
                    continue
                if limit_per_source and len(by_src[r.source]) >= limit_per_source:
                    break
                by_src[r.source].append(r)
    return dict(by_src)


class RecordSampler:
    """Weighted mixture over sources; each source is an epoch-shuffled cycle."""

    def __init__(self, by_src: Dict[str, List[Record]], weights: Dict[str, float], seed: int):
        self.rng = random.Random(seed)
        self.src = list(by_src)
        self.pools = {s: list(v) for s, v in by_src.items()}
        default_w = weights.get("__default__", 1.0)
        # weight ~ user weight * sqrt(size) so big sources don't dominate and small ones aren't starved
        self.w = [weights.get(s, default_w) * math.sqrt(len(self.pools[s])) for s in self.src]
        self.pos = {s: 0 for s in self.src}
        for s in self.src:
            self.rng.shuffle(self.pools[s])

    def next(self) -> Record:
        s = self.rng.choices(self.src, weights=self.w)[0]
        p = self.pools[s]
        if self.pos[s] >= len(p):
            self.rng.shuffle(p)
            self.pos[s] = 0
        r = p[self.pos[s]]
        self.pos[s] += 1
        return r


def encode_record(fmt: Formatter, r: Record, rng: random.Random, abstain_rate: float):
    qs = {k: parse_question(v) for k, v in r.questions.items()}
    drop = []
    for k, q in qs.items():
        if q.type == "choice" and len(q.criteria) >= 3 and rng.random() < abstain_rate:
            drop.append(k)
    indent = rng.choice([None, None, 1, 2])
    enc = fmt.encode(r.state, qs, labels=r.labels, shuffle_options=True, shuffle_questions=True,
                     drop_label_for=drop, rng=rng, state_indent=indent)
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init", default=None, help="decision checkpoint to continue from (else build from model_path causal LM)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--limit-per-source", type=int, default=0)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    cfg = TrainConfig.load(args.config, resume=args.resume, max_steps=args.max_steps, output_dir=args.output_dir)
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.dump(out / "config.yaml")
    device, dtype = device_and_dtype(cfg)
    log = Logger(out)
    ex = cfg.extra

    if args.init or cfg.resume:
        src = cfg.resume or args.init
        tok = AutoTokenizer.from_pretrained(src)
        model = OpenThaiSystemOneForDecision.from_pretrained(src, dtype=dtype if device.type == "cuda" else torch.float32)
    else:
        model, tok = OpenThaiSystemOneForDecision.from_causal_lm(cfg.model_path, torch_dtype=dtype if device.type == "cuda" else torch.float32)
    model.to(device)
    if cfg.gradient_checkpointing:
        model.model.gradient_checkpointing_enable()
    # calibration stage: optionally freeze everything but the head + temperatures
    if ex.get("freeze_backbone"):
        for p in model.model.parameters():
            p.requires_grad_(False)
    model.train()
    fmt = Formatter(tok, max_total_tokens=cfg.seq_len, max_state_tokens=min(cfg.seq_len // 2, 32768))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    by_src = load_records(cfg.data["train_globs"], cfg.data.get("weights", {}), args.limit_per_source)
    n_total = sum(len(v) for v in by_src.values())
    log.log(event="data", sources={k: len(v) for k, v in by_src.items()}, total=n_total)
    sampler = RecordSampler(by_src, cfg.data.get("weights", {}), cfg.seed)
    rng = random.Random(cfg.seed + 1)

    examples_per_step = cfg.micro_batch * cfg.grad_accum
    total_steps = cfg.max_steps or int(cfg.epochs * n_total / examples_per_step)
    opt = make_optimizer(model, cfg)
    step = 0
    if cfg.resume:
        st = torch.load(Path(cfg.resume) / "trainer_state.pt", map_location="cpu")
        opt.load_state_dict(st["optimizer"])
        step = st["step"]
    log.log(event="start", total_steps=total_steps, examples_per_step=examples_per_step, device=str(device))

    abstain_rate = float(ex.get("abstain_rate", 0.03))
    label_smoothing = float(ex.get("label_smoothing", 0.03))
    brier_weight = float(ex.get("brier_weight", 0.0))
    apply_temperature = bool(ex.get("train_temperature", False))

    t0 = time.time()
    while step < total_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps, cfg)
        loss_acc, n_q, n_correct = 0.0, 0, 0
        for _ in range(cfg.grad_accum):
            encs = [encode_record(fmt, sampler.next(), rng, abstain_rate) for _ in range(cfg.micro_batch)]
            batch = collate(encs, pad_id)
            qtypes = torch.full_like(batch["option_counts"], -1)
            for b, e in enumerate(encs):
                for qi, spec in enumerate(e.specs):
                    qtypes[b, qi] = QTYPE_INDEX[spec.qtype]
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                o = model(**batch, qtypes=qtypes.to(device), label_smoothing=label_smoothing,
                          brier_weight=brier_weight, apply_temperature=apply_temperature)
            (o.loss / cfg.grad_accum).backward()
            loss_acc += float(o.loss) / cfg.grad_accum
            lab = batch["labels"]
            keep = lab != -100
            n_q += int(keep.sum())
            n_correct += int((o.logits.argmax(-1)[keep] == lab[keep]).sum())
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % cfg.log_every == 0:
            el = time.time() - t0
            log.log(step=step, loss=loss_acc, acc=n_correct / max(1, n_q), lr=opt.param_groups[0]["lr"], grad_norm=float(gn),
                    ex_per_s=step * examples_per_step / el, eta_h=(total_steps - step) * el / step / 3600,
                    temps=[round(float(t), 3) for t in model.log_temperature.exp()])
        if step % cfg.save_every == 0 or step == total_steps:
            save_checkpoint(out, model, tok, opt, step)
    log.log(event="done", steps=step)


if __name__ == "__main__":
    main()
