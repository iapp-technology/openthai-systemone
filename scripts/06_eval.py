"""Stage 4: zero-shot evaluation on held-out decision datasets + calibration + latency.

    python scripts/06_eval.py --model runs/calib/latest --data data/decision --out runs/eval.json
    python scripts/06_eval.py --model runs/calib/latest --data data/decision --sources wisesight,banking77 --limit 500

Reports per source: accuracy, macro-F1 (choice), MAE (score), AUROC-ish via accuracy@threshold (noul), ECE, Brier.
Also measures batch-1 latency for a 255-option question.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openthai_systemone.client import SystemOneClient  # noqa: E402
from openthai_systemone.data.records import read_jsonl  # noqa: E402
from openthai_systemone.types import Choice  # noqa: E402


def ece(confs, corrects, bins=10):
    tot = len(confs)
    if tot == 0:
        return 0.0
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = sum(corrects[i] for i in idx) / len(idx)
        conf = sum(confs[i] for i in idx) / len(idx)
        e += len(idx) / tot * abs(acc - conf)
    return e


def macro_f1(y_true, y_pred):
    labels = set(y_true) | set(y_pred)
    f1s = []
    for l in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == l and p == l)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != l and p == l)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == l and p != l)
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def evaluate_source(client, records, batch_size, permutations=1):
    stats = defaultdict(list)
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        resps = client.system_one_batch([(r.state, r.questions) for r in chunk], permutations=permutations)
        for r, resp in zip(chunk, resps):
            for qid, ans in resp.answers.items():
                gold = r.labels.get(qid)
                if gold is None:
                    continue
                if ans.type == "choice":
                    correct = ans.choice == gold
                    stats["choice_true"].append(gold)
                    stats["choice_pred"].append(ans.choice)
                    stats["choice_correct"].append(1.0 if correct else 0.0)
                    stats["choice_conf"].append(max(ans.probabilities.values()))
                    stats["choice_brier"].append(sum((p - (1.0 if k == gold else 0.0)) ** 2 for k, p in ans.probabilities.items()))
                elif ans.type == "score":
                    pred = max(ans.probabilities, key=ans.probabilities.get)
                    stats["score_mae"].append(abs(ans.score - int(gold)))
                    stats["score_correct"].append(1.0 if int(pred) == int(gold) else 0.0)
                    stats["score_conf"].append(max(ans.probabilities.values()))
                elif ans.type == "noul":
                    pred = ans.noul >= 0.5
                    stats["noul_correct"].append(1.0 if pred == bool(gold) else 0.0)
                    stats["noul_conf"].append(max(ans.noul, 1 - ans.noul))
                    stats["noul_brier"].append((ans.noul - (1.0 if gold else 0.0)) ** 2)
    out = {}
    if stats["choice_correct"]:
        out["choice"] = {"n": len(stats["choice_correct"]), "acc": sum(stats["choice_correct"]) / len(stats["choice_correct"]),
                         "macro_f1": macro_f1(stats["choice_true"], stats["choice_pred"]),
                         "ece": ece(stats["choice_conf"], stats["choice_correct"]),
                         "brier": sum(stats["choice_brier"]) / len(stats["choice_brier"])}
    if stats["score_correct"]:
        out["score"] = {"n": len(stats["score_correct"]), "exact": sum(stats["score_correct"]) / len(stats["score_correct"]),
                        "mae": sum(stats["score_mae"]) / len(stats["score_mae"]), "ece": ece(stats["score_conf"], stats["score_correct"])}
    if stats["noul_correct"]:
        out["noul"] = {"n": len(stats["noul_correct"]), "acc": sum(stats["noul_correct"]) / len(stats["noul_correct"]),
                       "ece": ece(stats["noul_conf"], stats["noul_correct"]), "brier": sum(stats["noul_brier"]) / len(stats["noul_brier"])}
    return out


@torch.no_grad()
def latency(client, n_options=255, repeats=20):
    q = {"pick": Choice(instructions="Which product category fits the item?", criteria={f"category {i} (หมวด {i})": None for i in range(n_options)})}
    state = "สินค้า: เสื้อยืดคอกลมผ้าฝ้าย 100% สีขาว ไซส์ L ราคา 299 บาท ส่งฟรีเมื่อซื้อครบ 500 บาท"
    client.system_one(state, q)  # warmup
    if client.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        client.system_one(state, q)
    if client.device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/decision")
    ap.add_argument("--sources", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="")
    ap.add_argument("--permutations", type=int, default=1, help="option orders averaged per choice question (1 = single order)")
    args = ap.parse_args()
    client = SystemOneClient(args.model)
    files = sorted(glob.glob(str(Path(args.data) / "*.eval.jsonl")))
    want = set(s for s in args.sources.split(",") if s)
    results = {}
    for f in files:
        name = Path(f).name.split(".")[0]
        if want and name not in want:
            continue
        recs = list(read_jsonl(f))
        if args.limit:
            recs = recs[: args.limit]
        t0 = time.time()
        results[name] = evaluate_source(client, recs, args.batch_size, args.permutations)
        results[name]["seconds"] = round(time.time() - t0, 1)
        print(name, json.dumps(results[name], ensure_ascii=False), flush=True)
    results["_latency_ms_255_options_batch1"] = latency(client)
    results["_device"] = client.device
    results["_permutations"] = args.permutations
    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
