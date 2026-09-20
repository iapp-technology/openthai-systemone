"""Stage 5: package a trained checkpoint for the Hub (trust_remote_code files + model card) and optionally push.

    python scripts/07_export_release.py --ckpt runs/calib/latest --out release/OpenThai-SystemOne --eval runs/eval.json
    python scripts/07_export_release.py --ckpt runs/calib/latest --out release/OpenThai-SystemOne --push iapp/OpenThai-SystemOne

The exported folder loads with:
    AutoModel.from_pretrained("iapp/OpenThai-SystemOne", trust_remote_code=True)
or via the pip package (no remote code needed):
    from openthai_systemone import SystemOneClient
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PKG = Path(__file__).resolve().parents[1] / "openthai_systemone"


NIMBLE = {  # Bespoke Labs, docs/PUBLIC_BENCHMARKS.md (2026-09-18): subset -> (Nimble-9B, Jev 1.13.0)
    "aegis2": (81.2, 80.4), "boolq": (86.0, 89.7), "civil_comments": (70.3, 81.0), "helpsteer2": (39.0, 34.1),
    "massive-en-US": (86.9, 87.4), "massive-de-DE": (83.4, 86.9), "multinli": (85.3, 82.9), "paws": (82.8, 89.2),
    "pubmedqa": (75.6, 77.2), "squad2": (80.6, 82.9), "summeval-consistency": (75.7, 81.2), "summeval-relevance": (49.2, 35.0),
    "vitaminc-dev": (76.6, 80.1),
}
THAI_SETS = {  # source -> (label, held-out note)
    "wisesight": ("Wisesight sentiment (4-class)", "whole dataset held out"), "sib200_th": ("SIB-200 Thai topic (7-class)", "whole dataset held out"),
    "contrastive_th": ("Thai contrastive pairs (one-fact flips)", "synthetic, eval-only"), "xnli_th": ("XNLI-th", "eval split"),
    "massive_th": ("MASSIVE-th intent", "eval split"), "prachathai": ("Prachathai topics", "eval split"), "wongnai": ("Wongnai stars (score)", "eval split"),
    "banking77": ("banking77 intent (77-way, English)", "whole dataset held out"), "xlam_tools": ("xLAM tool selection (English)", "eval slice"),
}


def _main_metric(r):
    return r.get("acc", r.get("exact"))


def render_card(eval_public: dict | None, eval_heldout: dict | None) -> str:
    pub = "_pending_"
    if eval_public:
        rows, ours, theirs_n, theirs_j = [], [], [], []
        for name, (n9, j) in NIMBLE.items():
            res = eval_public.get(name) or eval_public.get(f"pub_{name}")
            if not res:
                continue
            r = next(v for k, v in res.items() if isinstance(v, dict) and "n" in v)
            a = 100 * _main_metric(r)
            ours.append(a); theirs_n.append(n9); theirs_j.append(j)
            rows.append(f"| {name} | {r['n']} | **{a:.1f}** | {n9:.1f} | {j:.1f} | {r['ece']:.3f} |")
        if ours:
            rows.append(f"| **macro average** | | **{sum(ours)/len(ours):.1f}** | {sum(theirs_n)/len(theirs_n):.1f} | {sum(theirs_j)/len(theirs_j):.1f} | |")
            pub = "| subset | n | OpenThai-SystemOne 0.8B | Bespoke-Nimble-9B | Jev 1.13.0 | our ECE |\n|---|---|---|---|---|---|\n" + "\n".join(rows)
    thai = "_pending_"
    if eval_heldout:
        rows = []
        for src, (label, note) in THAI_SETS.items():
            res = eval_heldout.get(src)
            if not res:
                continue
            for qt in ("choice", "noul", "score"):
                if qt in res:
                    r = res[qt]
                    extra = f" (MAE {r['mae']:.2f})" if qt == "score" else (f" (macro-F1 {r['macro_f1']:.3f})" if "macro_f1" in r else "")
                    rows.append(f"| {label} | {qt} | {r['n']} | **{100*_main_metric(r):.1f}**{extra} | {r['ece']:.3f} | {note} |")
        if rows:
            thai = "| set | type | n | accuracy | ECE | note |\n|---|---|---|---|---|---|\n" + "\n".join(rows)
    lat = ""
    for ev in (eval_public, eval_heldout):
        if ev and ev.get("_latency_ms_255_options_batch1"):
            lat = f"Batch-1 latency, one question with 255 options, {ev.get('_device')}: **{ev['_latency_ms_255_options_batch1']:.0f} ms**"
    card = Path(__file__).resolve().parents[1] / "docs" / "MODEL_CARD.md"
    text = card.read_text() if card.exists() else "# OpenThai-SystemOne\n"
    return text.replace("{{PUBLIC_TABLE}}", pub).replace("{{THAI_TABLE}}", thai).replace("{{LATENCY}}", lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-public", default="", help="runs/eval_public.json")
    ap.add_argument("--eval-heldout", default="", help="runs/eval_heldout.json")
    ap.add_argument("--push", default="", help="repo id, e.g. iapp/OpenThai-SystemOne")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.ckpt).resolve(), Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in src.iterdir():
        if f.name in ("trainer_state.pt",):
            continue
        shutil.copy2(f, out / f.name)

    # ship remote code so AutoModel works without the pip package
    for name in ("configuration.py", "modeling.py", "formatting.py", "types.py"):
        shutil.copy2(PKG / name, out / f"{name}")
    cfg = json.loads((out / "config.json").read_text())
    # checkpoints trained before the 2026-09-20 rename carry the old model_type / architecture name
    cfg["model_type"] = "openthai_systemone"
    cfg["architectures"] = ["OpenThaiSystemOneForDecision"]
    cfg["auto_map"] = {
        "AutoConfig": "configuration.OpenThaiSystemOneConfig",
        "AutoModel": "modeling.OpenThaiSystemOneForDecision",
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    # relative imports (from .formatting import ...) are resolved by transformers' dynamic module loader

    evp = json.loads(Path(args.eval_public).read_text()) if args.eval_public else None
    evh = json.loads(Path(args.eval_heldout).read_text()) if args.eval_heldout else None
    (out / "README.md").write_text(render_card(evp, evh))
    assets = Path(__file__).resolve().parents[1] / "docs" / "assets"
    if assets.is_dir():
        shutil.copytree(assets, out / "assets", dirs_exist_ok=True)
    print("exported ->", out)

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, exist_ok=True, private=args.private, repo_type="model")
        api.upload_folder(folder_path=str(out), repo_id=args.push, commit_message="OpenThai-SystemOne release")
        print("pushed ->", f"https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
