"""Final CPT evaluation: base vs CPT on the held-out mixture + two fixed probes. Writes runs/cpt/final_eval.json."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
cpt = importlib.import_module("02_cpt_train")
prep = importlib.import_module("00_prepare_base")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="base/qwen3.5-0.8b-text")
    ap.add_argument("--ckpt", default="runs/cpt/latest")
    ap.add_argument("--eval-bin", default="data/cpt/eval.bin")
    ap.add_argument("--out", default="runs/cpt/final_eval.json")
    args = ap.parse_args()
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.base)
    res = {"ppl": {}, "probes": {}}
    losses = {}
    for name, path in (("base", args.base), ("cpt", args.ckpt)):
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
        e = cpt.evaluate(m, args.eval_bin, 4096, dev, torch.bfloat16, max_batches=200)
        res["ppl"][name] = e["eval_ppl"]
        losses[name] = {"thai_probe": prep.lm_loss(m, tok, prep.THAI_SAMPLE, dev), "english_probe": prep.lm_loss(m, tok, prep.EN_SAMPLE, dev)}
        del m
        torch.cuda.empty_cache()
    for probe in ("thai_probe", "english_probe"):
        res["probes"][probe] = {"base": losses["base"][probe], "cpt": losses["cpt"][probe]}
    # intermediate evals from the training log
    log = Path(args.ckpt).resolve().parent / "train_log.jsonl"
    if log.exists():
        for line in open(log):
            r = json.loads(line)
            if "eval_ppl" in r:
                res["ppl"][f"step {r['step']}"] = r["eval_ppl"]
    res["ppl"] = {"base model": res["ppl"].pop("base"), **{k: v for k, v in res["ppl"].items() if k != "cpt"}, "final (CPT)": res["ppl"].pop("cpt")}
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
