"""Stage 1: Thai continued pretraining of the text-only Qwen3.5-0.8B (standard next-token loss).

    python scripts/02_cpt_train.py --config configs/cpt.yaml
    python scripts/02_cpt_train.py --config configs/cpt.yaml --resume runs/cpt/latest

Single GPU, bf16, packed fixed-length sequences from data/cpt/shard-*.bin, cosine LR, checkpoints every N steps.
"""
from __future__ import annotations

import argparse
import glob
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openthai_systemone.train_utils import (  # noqa: E402
    Logger, TrainConfig, device_and_dtype, lr_at, make_optimizer, save_checkpoint, set_seed,
)


class PackedShards:
    """Contiguous uint32 token stream -> (seq_len+1)-length windows, shuffled by block."""

    def __init__(self, pattern: str, seq_len: int, seed: int = 0):
        self.files = sorted(glob.glob(pattern))
        assert self.files, f"no shards match {pattern}"
        self.arrs = [np.memmap(f, dtype=np.uint32, mode="r") for f in self.files]
        self.seq_len = seq_len
        self.windows = []
        for fi, a in enumerate(self.arrs):
            n = (len(a) - 1) // seq_len
            self.windows.extend((fi, i) for i in range(n))
        rng = np.random.default_rng(seed)
        rng.shuffle(self.windows)

    def __len__(self):
        return len(self.windows)

    def batch(self, idx: int, bs: int):
        xs = []
        for fi, i in self.windows[idx * bs : (idx + 1) * bs]:
            a = self.arrs[fi][i * self.seq_len : (i + 1) * self.seq_len + 1]
            xs.append(torch.from_numpy(a.astype(np.int64)))
        x = torch.stack(xs)
        return x[:, :-1], x[:, 1:]


def chunked_lm_loss(model, x: torch.Tensor, y: torch.Tensor, chunk: int = 1024) -> torch.Tensor:
    """Next-token loss without materialising (B*T, vocab) logits: the 248k vocab makes that ~30 GB per 32k tokens.

    Runs the body once, then applies lm_head + CE per chunk of tokens under activation checkpointing, so only one
    chunk of logits exists at a time in forward and in backward.
    """
    from torch.utils.checkpoint import checkpoint

    base = model.model if hasattr(model, "model") else model.base_model
    hidden = base(input_ids=x).last_hidden_state  # (B, T, H)
    head = model.lm_head
    B, T, H = hidden.shape
    h = hidden.reshape(B * T, H)
    t = y.reshape(B * T)
    n = t.numel()

    def chunk_loss(hc, tc):
        return torch.nn.functional.cross_entropy(head(hc).float(), tc, reduction="sum")

    total = hidden.new_zeros((), dtype=torch.float32)
    for i in range(0, n, chunk):
        total = total + checkpoint(chunk_loss, h[i : i + chunk], t[i : i + chunk], use_reentrant=False)
    return total / n


@torch.no_grad()
def evaluate(model, eval_path: str, seq_len: int, device, dtype, max_batches: int = 32):
    a = np.memmap(eval_path, dtype=np.uint32, mode="r")
    n = min(max_batches, (len(a) - 1) // seq_len)
    losses = []
    model.eval()
    for i in range(n):
        x = torch.from_numpy(a[i * seq_len : (i + 1) * seq_len + 1].astype(np.int64))[None].to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            loss = chunked_lm_loss(model, x[:, :-1], x[:, 1:])
        losses.append(float(loss))
    model.train()
    m = sum(losses) / max(1, len(losses))
    return {"eval_loss": m, "eval_ppl": math.exp(m)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    cfg = TrainConfig.load(args.config, resume=args.resume, max_steps=args.max_steps, output_dir=args.output_dir)
    set_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg.dump(out / "config.yaml")
    device, dtype = device_and_dtype(cfg)
    log = Logger(out)

    tok = AutoTokenizer.from_pretrained(cfg.model_path)
    src = cfg.resume or cfg.model_path
    model = AutoModelForCausalLM.from_pretrained(src, dtype=dtype if device.type == "cuda" else torch.float32).to(device)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()

    data = PackedShards(cfg.data["train_glob"], cfg.seq_len, cfg.seed)
    tokens_per_step = cfg.micro_batch * cfg.grad_accum * cfg.seq_len
    total_steps = cfg.max_steps or int(cfg.total_tokens // tokens_per_step)
    total_steps = min(total_steps, len(data) // (cfg.micro_batch * cfg.grad_accum))
    log.log(event="start", windows=len(data), tokens_per_step=tokens_per_step, total_steps=total_steps, device=str(device))

    opt = make_optimizer(model, cfg)
    step = 0
    if cfg.resume:
        st = torch.load(Path(cfg.resume) / "trainer_state.pt", map_location="cpu")
        opt.load_state_dict(st["optimizer"])
        step = st["step"]
    if cfg.compile and device.type == "cuda":
        model = torch.compile(model)

    t0 = time.time()
    seen = 0
    step0 = step  # steps completed before this (re)start; ETA is based on steps done in this process
    while step < total_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps, cfg)
        loss_acc = 0.0
        for micro in range(cfg.grad_accum):
            idx = step * cfg.grad_accum + micro
            x, y = data.batch(idx, cfg.micro_batch)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                loss = chunked_lm_loss(model, x, y) / cfg.grad_accum
            loss.backward()
            loss_acc += float(loss)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        seen += tokens_per_step
        if step % cfg.log_every == 0:
            el = time.time() - t0
            log.log(step=step, loss=loss_acc, lr=opt.param_groups[0]["lr"], grad_norm=float(gn),
                    tok_per_s=seen / el, eta_h=(total_steps - step) * el / max(1, step - step0) / 3600)
        if step % cfg.eval_every == 0 and cfg.data.get("eval_path"):
            log.log(step=step, **evaluate(model, cfg.data["eval_path"], cfg.seq_len, device, dtype))
        if step % cfg.save_every == 0 or step == total_steps:
            save_checkpoint(out, model, tok, opt, step)
    log.log(event="done", steps=step, tokens=seen)


if __name__ == "__main__":
    main()
