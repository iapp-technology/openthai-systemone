"""Build the 13 public benchmark subsets used by Bespoke Nimble (docs/PUBLIC_BENCHMARKS.md) as eval records.

    python scripts/06b_public_benchmarks.py --out data/public_bench [--tokenizer base/qwen3.5-0.8b-text]
    python scripts/06_eval.py --model runs/sft/latest --data data/public_bench --out runs/eval_public.json

Same sources, splits, instructions, criteria, per-subset limits, stratified whole-family sampling (seed 20260918)
and 2048-token input filter as Nimble's `nimble.datasets.public_benchmarks`, so numbers are comparable (the exact
sampled ids can differ slightly because the length filter uses our tokenizer/prompt). Instruction and criteria
strings come from `configs/nimble_public_bench.json`, extracted from the Nimble repo (Bespoke Labs, 2026).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openthai_systemone.data.records import Record, write_jsonl  # noqa: E402
from openthai_systemone.types import parse_question  # noqa: E402

DEFS = json.loads((Path(__file__).resolve().parents[1] / "configs" / "nimble_public_bench.json").read_text())
SEED = 20260918
MAX_INPUT_TOKENS = 2048

# subset -> (limit, hf dataset, config, split)
SUBSETS = {
    "vitaminc-dev": (600, "tals/vitaminc", None, "validation"),
    "massive-en-US": (350, "mteb/amazon_massive_scenario", "en", "test"),
    "massive-de-DE": (350, "mteb/amazon_massive_scenario", "de", "test"),
    "boolq": (300, "google/boolq", None, "validation"),
    "squad2": (300, "rajpurkar/squad_v2", None, "validation"),
    "paws": (250, "google-research-datasets/paws", "labeled_final", "test"),
    "multinli": (300, "nyu-mll/multi_nli", None, "validation_matched"),
    "civil_comments": (300, "google/civil_comments", None, "test"),
    "aegis2": (250, "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", None, "test"),
    "helpsteer2": (250, "nvidia/HelpSteer2", None, "validation"),
    "summeval-relevance": (240, "mteb/summeval", None, "test"),
    "summeval-consistency": (150, "mteb/summeval", None, "test"),
    "pubmedqa": (250, "qiaojin/PubMedQA", "pqa_labeled", "train"),
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def q_choice(d, inst, target):
    return {"type": "choice", "instructions": inst, "criteria": dict(d)}, target


def q_noul(crit, inst, target: bool):
    return {"type": "noul", "instructions": inst, "criteria": {"false": crit["false"], "true": crit["true"]}}, bool(target)


def q_score(levels, inst, target: int):
    return {"type": "score", "instructions": inst, "criteria": list(levels)}, int(target)


def convert(subset: str, ex: dict):
    """Return (id, domain, family, state, question, label) or None to skip."""
    if subset == "boolq":
        d = DEFS["boolq"]
        if type(ex.get("answer")) is not bool:
            return None
        q, t = q_noul(d["CRITERIA"], d["INSTRUCTIONS"], ex["answer"])
        return ("boolq-" + digest(ex["question"] + "\n" + ex["passage"]), "boolq", digest(ex["passage"]),
                {"passage": ex["passage"], "question": ex["question"]}, q, t)
    if subset == "multinli":
        d = DEFS["multinli"]
        if ex["label"] not in (0, 1, 2):
            return None
        lab = d["OPTIONS"][ex["label"]]
        q, t = q_choice(d["CRITERIA"], d["INSTRUCTIONS"], lab)
        return ("multinli-" + str(ex["pairID"]), "multinli-" + ex["genre"], str(ex["promptID"]),
                {"premise": ex["premise"].strip(), "hypothesis": ex["hypothesis"].strip()}, q, t)
    if subset == "paws":
        d = DEFS["paws"]
        q, t = q_noul(d["CRITERIA"], d["INSTRUCTIONS"], ex["label"] == 1)
        i = str(ex["id"])
        return ("paws-" + i, "paws", i, {"sentence_1": ex["sentence1"], "sentence_2": ex["sentence2"]}, q, t)
    if subset == "squad2":
        d = DEFS["squad2"]
        texts = sorted(set(t for t in ex["answers"]["text"] if t.strip()))
        q, t = q_noul(d["CRITERIA"], d["INSTRUCTIONS"], bool(texts))
        return ("squad2-" + ex["id"], "squad2", digest(ex["context"]), {"paragraph": ex["context"], "question": ex["question"]}, q, t)
    if subset == "civil_comments":
        d = DEFS["civil_comments"]
        if not ex["text"].strip():
            return None
        q, t = q_noul(d["CRITERIA"], d["INSTRUCTIONS"], ex["toxicity"] >= d["THRESHOLD"])
        i = f"civil_comments-{ex['_row']}"
        return (i, "civil_comments", i, ex["text"].strip(), q, t)
    if subset == "aegis2":
        d = DEFS["aegis2"]
        if ex.get("prompt_label_source") != "human" or ex.get("reconstruction_id_if_redacted") is not None:
            return None
        if not isinstance(ex.get("prompt"), str) or not ex["prompt"].strip() or ex.get("prompt_label") not in d["LABELS"]:
            return None
        q, t = q_noul(d["CRITERIA"], d["INSTRUCTIONS"], d["LABELS"][ex["prompt_label"]])
        i = "aegis2-" + ex["id"]
        return (i, "aegis2", i, {"user_message": ex["prompt"].strip()}, q, t)
    if subset == "helpsteer2":
        d = DEFS["helpsteer2"]
        q, t = q_score(d["LEVELS"], d["INSTRUCTIONS"], int(ex["helpfulness"]))
        return (f"helpsteer2-{ex['_row']}", "helpsteer2", "helpsteer2-" + digest(ex["prompt"]),
                {"prompt": ex["prompt"], "response": ex["response"]}, q, t)
    if subset.startswith("summeval-"):
        dim = subset.split("-", 1)[1]
        d = DEFS["summeval"]
        mean = ex["_mean"]
        level = min(4, max(0, math.floor(mean + 0.5) - 1))
        q, t = q_score(d["LEVELS"][dim], d["INSTRUCTIONS"][dim], level)
        return (f"summeval-{dim}-{ex['id']}-{ex['_k']}", "summeval-" + dim, str(ex["id"]),
                {"article": ex["text"], "summary": ex["_summary"]}, q, t)
    if subset == "pubmedqa":
        d = DEFS["pubmedqa"]
        dec = (ex.get("final_decision") or "").strip().lower()
        ctx = ex["context"]["contexts"] if isinstance(ex["context"], dict) else ex["context"]
        ctx = [c for c in (ctx or []) if isinstance(c, str) and c.strip()]
        if dec not in d["CRITERIA"] or not ctx:
            return None
        q, t = q_choice(d["CRITERIA"], d["INSTRUCTIONS"], dec)
        p = str(ex["pubid"])
        return ("pubmedqa-" + p, "pubmedqa", p, {"question": ex["question"], "abstract_context": " ".join(ctx)}, q, t)
    if subset == "vitaminc-dev":
        d = DEFS["vitaminc"]
        if ex["label"] not in d["CRITERIA"]:
            return None
        q, t = q_choice(d["CRITERIA"], d["INSTRUCTIONS"], ex["label"])
        return ("vitaminc-" + str(ex["unique_id"]), "vitaminc-" + ex["revision_type"], str(ex["case_id"]),
                {"evidence": ex["evidence"], "claim": ex["claim"]}, q, t)
    if subset.startswith("massive-"):
        loc = subset.split("-", 1)[1]
        d = DEFS["massive"]
        scen = ex.get("label_text") or ex.get("label")
        if scen not in d["CRITERIA"] or not ex["text"].strip():
            return None
        q, t = q_choice(d["CRITERIA"], d["INSTRUCTIONS"], scen)
        i = f"massive-{ex['id']}"
        return (i, f"massive-{loc}", i, {"utterance": ex["text"], "locale": loc}, q, t)
    raise ValueError(subset)


def rows_for(subset: str):
    from datasets import load_dataset

    _, name, cfg, split = SUBSETS[subset]
    ds = load_dataset(name, cfg, split=split)
    if subset.startswith("summeval-"):
        dim = subset.split("-", 1)[1]
        for ex in ds:
            for k, summ in enumerate(ex["machine_summaries"]):
                yield {**ex, "_k": k, "_summary": summ, "_mean": ex[dim][k]}
        return
    for i, ex in enumerate(ds):
        yield {**ex, "_row": i}


def select(records, seed, limit):
    """Nimble's stratified whole-family sampler (metadata only)."""
    families = {}
    for r in records:
        families.setdefault(r["family"], []).append(r)
    strata = {}
    for fam, group in families.items():
        strata.setdefault(group[0]["domain"], []).append(fam)
    rng = random.Random(seed)
    order = []
    for domain in sorted(strata):
        keys = sorted(strata[domain])
        rng.shuffle(keys)
        order.append(keys)
    selected, index, stop = [], 0, False
    while not stop and any(index < len(keys) for keys in order):
        for keys in order:
            if index >= len(keys):
                continue
            group = families[keys[index]]
            if limit is not None and len(selected) + len(group) > limit:
                stop = True
                break
            selected.extend(group)
        index += 1
    selected.sort(key=lambda r: r["id"])
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/public_bench")
    ap.add_argument("--only", default="")
    ap.add_argument("--tokenizer", default="base/qwen3.5-0.8b-text")
    ap.add_argument("--no-length-filter", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    measure = None
    if not args.no_length_filter:
        from transformers import AutoTokenizer

        from openthai_systemone.formatting import Formatter

        fmt = Formatter(AutoTokenizer.from_pretrained(args.tokenizer))
        measure = lambda state, q: fmt.encode(state, {"decision": parse_question(q)}).n_tokens  # noqa: E731

    names = [n for n in args.only.split(",") if n] or list(SUBSETS)
    manifest = {}
    for subset in names:
        limit = SUBSETS[subset][0]
        conv, dropped, skipped = [], 0, 0
        for ex in rows_for(subset):
            r = convert(subset, ex)
            if r is None:
                skipped += 1
                continue
            rid, domain, family, state, q, t = r
            if measure is not None and measure(state, q) > MAX_INPUT_TOKENS:
                dropped += 1
                continue
            conv.append({"id": rid, "domain": domain, "family": family, "state": state, "q": q, "t": t})
        sel = select(conv, SEED, limit)
        recs = [Record(r["id"], f"pub_{subset}", "de" if "de-DE" in subset else "en", r["state"], {"decision": r["q"]},
                       {"decision": r["t"]}, split="eval", meta={"domain": r["domain"], "family": r["family"]}) for r in sel]
        n = write_jsonl(out / f"{subset}.eval.jsonl", recs)
        manifest[subset] = {"count": n, "limit": limit, "converted": len(conv), "skipped": skipped, "dropped_over_length": dropped,
                            "types": dict(Counter(r["q"]["type"] for r in sel)), "labels": dict(Counter(str(r["t"]) for r in sel))}
        print(subset, manifest[subset], flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
