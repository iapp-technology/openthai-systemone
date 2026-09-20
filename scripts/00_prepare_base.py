"""Stage 0: turn Qwen/Qwen3.5-0.8B-Base (vision + text) into a text-only causal LM checkpoint.

    python scripts/00_prepare_base.py --src base/Qwen3.5-0.8B-Base --dst base/qwen3.5-0.8b-text

Checks that the text-only model reproduces the original's next-token loss on Thai + English samples.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

THAI_SAMPLE = (
    "กรุงเทพมหานครเป็นเมืองหลวงและเมืองที่มีประชากรมากที่สุดของประเทศไทย "
    "ตั้งอยู่ริมแม่น้ำเจ้าพระยา มีบทบาทสำคัญทางเศรษฐกิจ การเมือง และวัฒนธรรมของประเทศ "
    "ลูกค้าแจ้งว่าไม่สามารถเข้าสู่ระบบได้ตั้งแต่เมื่อวานนี้ และต้องการขอเงินคืนภายในสัปดาห์นี้"
)
EN_SAMPLE = (
    "Bangkok is the capital and most populous city of Thailand. The customer reports that they cannot "
    "log in since yesterday and would like a refund within the week. Click the 'Submit' button to continue."
)


@torch.no_grad()
def lm_loss(model, tok, text, device):
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    out = model(input_ids=ids, labels=ids)
    return float(out.loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="base/Qwen3.5-0.8B-Base")
    ap.add_argument("--dst", default="base/qwen3.5-0.8b-text")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    tok = AutoTokenizer.from_pretrained(src)

    print("loading multimodal checkpoint…")
    t0 = time.time()
    mm = AutoModelForImageTextToText.from_pretrained(src, dtype=dtype)
    print(f"  loaded in {time.time() - t0:.1f}s; params={sum(p.numel() for p in mm.parameters()) / 1e6:.1f}M")

    # locate the text tower and the LM head
    lm_head = mm.lm_head
    text_model = None
    for name in ("language_model", "model.language_model", "model"):
        obj = mm
        ok = True
        for part in name.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok and hasattr(obj, "config") and getattr(obj.config, "model_type", "") == "qwen3_5_text":
            text_model = obj
            break
    if text_model is None:
        raise SystemExit("could not locate the qwen3_5_text tower; print(mm) and adjust")

    text_cfg: AutoConfig = text_model.config
    text_cfg.tie_word_embeddings = True
    text_cfg.architectures = ["Qwen3_5ForCausalLM"]
    lm = AutoModelForCausalLM.from_config(text_cfg, dtype=dtype)
    missing, unexpected = lm.model.load_state_dict(text_model.state_dict(), strict=False)
    print("  missing:", missing, "unexpected:", unexpected)
    assert not unexpected and not missing, "text tower state dict mismatch"
    lm.lm_head.weight = lm.get_input_embeddings().weight if text_cfg.tie_word_embeddings else lm_head.weight
    if not torch.equal(lm.lm_head.weight, lm_head.weight):
        lm.lm_head.weight.data.copy_(lm_head.weight.data)
    n_params = sum(p.numel() for p in lm.parameters())
    print(f"  text-only params={n_params / 1e6:.1f}M (embedding={lm.get_input_embeddings().weight.numel() / 1e6:.1f}M)")

    if not args.skip_check:
        mm.to(args.device).eval()
        lm.to(args.device).eval()
        report = {}
        for name, text in [("thai", THAI_SAMPLE), ("english", EN_SAMPLE)]:
            a = lm_loss(mm, tok, text, args.device)
            b = lm_loss(lm, tok, text, args.device)
            n_tok = len(tok(text).input_ids)
            report[name] = {"orig_loss": a, "text_only_loss": b, "tokens": n_tok, "chars_per_token": len(text) / n_tok}
            print(f"  {name:8s} orig={a:.4f} text-only={b:.4f} tokens={n_tok} chars/tok={len(text) / n_tok:.2f}")
            assert abs(a - b) < 1e-2, "text-only model diverges from original"
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "prepare_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    dst.mkdir(parents=True, exist_ok=True)
    lm.save_pretrained(dst, safe_serialization=True)
    tok.save_pretrained(dst)
    print("saved ->", dst)


if __name__ == "__main__":
    main()
