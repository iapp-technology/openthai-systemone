"""Rewrite the live model card's tables from a new pair of eval files (keeps prose, images, sponsor block)."""
import json, re, sys
src, out, pub_json, held_json = sys.argv[1:5]
VERSION = sys.argv[5] if len(sys.argv) > 5 else "v0.2"
DATE = sys.argv[6] if len(sys.argv) > 6 else "2026-09-21"
CHANGE = sys.argv[7] if len(sys.argv) > 7 else "+3,000 SFT steps from v0.1 with a 22k-record synthetic Thai social-sentiment set (4/3/5-class, yes/no, score schemes), re-calibrated"
TEMPS = sys.argv[8] if len(sys.argv) > 8 else "choice 1.055, noul 1.047, score 1.000"
CHANGELOG = sys.argv[9] if len(sys.argv) > 9 else ""
s = open(src, encoding="utf-8").read()
P = json.load(open(pub_json)); H = json.load(open(held_json))
NIM = {"aegis2": (81.2, 80.4), "boolq": (86.0, 89.7), "civil_comments": (70.3, 81.0), "helpsteer2": (39.0, 34.1),
       "massive-en-US": (86.9, 87.4), "massive-de-DE": (83.4, 86.9), "multinli": (85.3, 82.9), "paws": (82.8, 89.2),
       "pubmedqa": (75.6, 77.2), "squad2": (80.6, 82.9), "summeval-consistency": (75.7, 81.2), "summeval-relevance": (49.2, 35.0),
       "vitaminc-dev": (76.6, 80.1)}
def m(res):
    return next((k, v) for k, v in res.items() if isinstance(v, dict) and "n" in v)
ours = []
for sub, (n9, jv) in NIM.items():
    qt, r = m(P[sub]); acc = 100 * r.get("acc", r.get("exact")); ours.append(acc)
    cell = f"**{acc:.1f}**" if acc > n9 else f"{acc:.1f}"
    new = f"| {sub} | {qt} | {r['n']} | {cell} | {n9:.1f} | {jv:.1f} | {r['ece']:.3f} |"
    s, k = re.subn(rf"^\| {re.escape(sub)} \| \w+ \| \d+ \|.*$", new, s, count=1, flags=re.M); assert k == 1, sub
macro = sum(ours) / len(ours)
s, k = re.subn(r"^\| \*\*macro average\*\* \|.*$", f"| **macro average** | | | **{macro:.1f}** | 74.8 | 76.0 | |", s, flags=re.M); assert k == 1
s = s.replace("Reference: raw Qwen3.5-0.8B prompted with letter log-probs scores 45.4 macro on the same bench (Bespoke's number).",
              "For scale: Bespoke reports raw Qwen3.5-0.8B at 45.4 on their *private* 324-item holdout (not this bench), Nimble-9B at 90.1, Jev at 93.2.")
THAI = [("MASSIVE-th intent (60-way)", "massive_th", "choice"), ("Prachathai67k topics", "prachathai", "choice"), ("Prachathai67k topics", "prachathai", "noul"),
        ("XNLI-th", "xnli_th", "choice"), ("XNLI-th (entailment yes/no)", "xnli_th", "noul"), ("SIB-200 Thai topic (7-way)", "sib200_th", "choice"),
        ("Thai contrastive pairs (one-fact flips)", "contrastive_th", "choice"), ("Thai contrastive pairs", "contrastive_th", "score"), ("Thai contrastive pairs", "contrastive_th", "noul"),
        ("Wongnai review stars (1–5)", "wongnai", "score"), ("Wisesight sentiment (4-class)", "wisesight", "choice"), ("banking77 intent (77-way, English)", "banking77", "choice"),
        ("xLAM tool selection (English)", "xlam_tools", "choice")]
NOTES = {"wisesight": "whole dataset held out — v0.1 38.7 → v0.2 51.5 → now {acc:.1f} (weakest Thai set; use order-invariant mode)",
         "banking77": "whole dataset held out — 77-way near-duplicate intents (v0.1 32.7; 61.7 with order-invariant mode on v0.2)", "sib200_th": "whole dataset held out (v0.1: 77.5)"}
for label, src_key, qt in THAI:
    r = H[src_key][qt]; acc = 100 * r.get("acc", r.get("exact"))
    extra = f"MAE {r['mae']:.2f}" if qt == "score" else (f"F1 {r['macro_f1']:.3f}" if "macro_f1" in r else "")
    cell = f"**{acc:.1f}**" if acc >= 60 else f"{acc:.1f}"
    pat = rf"^\| {re.escape(label)} \| {qt} \| \d+ \| [^|]+\| [^|]*\| [^|]+\| (?P<note>[^|]+)\|$"
    mo = re.search(pat, s, flags=re.M); assert mo, (label, qt)
    note = NOTES.get(src_key, mo.group("note").strip()).format(acc=acc) if src_key in NOTES else mo.group("note").strip()
    s = s[:mo.start()] + f"| {label} | {qt} | {r['n']} | {cell} | {extra} | {r['ece']:.3f} | {note} |" + s[mo.end():]
s = re.sub(r"### Calibration \(Stage 3\)\n\n(v0\.\d+ learned temperatures:[^\n]*\n[^\n]*\n)?", f"### Calibration (Stage 3)\n\n{VERSION} learned temperatures: {TEMPS} (the before/after table below was measured on v0.1;\nthe procedure is identical in every version).\n", s, count=1)
s = re.sub(r"\* v0\.\d+ known weak spots \(numbers above\):.*?(?=\n\* |\n\n)", f"""* {VERSION} known weak spots (numbers above): summary *relevance* scoring ({100*m(P['summeval-relevance'])[1]['exact']:.0f}) and helpfulness
  scoring ({100*m(P['helpsteer2'])[1]['exact']:.0f}) — the two 5-level rating tasks; fine-grained 77-way English intents (banking77 {100*H['banking77']['choice']['acc']:.0f} single-order); Thai social
  sentiment (wisesight {100*H['wisesight']['choice']['acc']:.0f}); PubMedQA 3-way ({100*m(P['pubmedqa'])[1]['acc']:.0f}).""", s, count=1, flags=re.S)
row = f"| {VERSION} | {DATE} | {CHANGE} | **{macro:.1f}** | {100*H['wisesight']['choice']['acc']:.1f} |"
if "## Versions" not in s:
    s = s.replace("## License & credits", f"## Versions\n\n| version | date | change | public macro | Wisesight |\n|---|---|---|---|---|\n{row}\n| v0.1 | 2026-09-20 | initial release: Thai CPT 4.47B tokens, 12k-step SFT, calibration | 61.9 | 38.7 |\n\n## License & credits", 1)
else:
    s = re.sub(r"(\|---\|---\|---\|---\|---\|\n)(\| v0\.\d+ \|[^\n]*\n)", lambda mo: mo.group(1) + ("" if mo.group(2).startswith(f"| {VERSION} ") else row + "\n") + mo.group(2), s, count=1)
if CHANGELOG and "## Changelog" in s and f"**{VERSION} —" not in s:
    s = s.replace("## Changelog\n\n", "## Changelog\n\n" + CHANGELOG.rstrip() + "\n\n", 1)
open(out, "w", encoding="utf-8").write(s)
print(f"macro {macro:.1f}; card written -> {out}")
