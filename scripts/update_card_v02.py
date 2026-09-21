"""Rewrite the live model card's tables from a new pair of eval files (keeps prose, images, sponsor block)."""
import json, re, sys
src, out, pub_json, held_json = sys.argv[1:5]
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
NOTES = {"wisesight": "whole dataset held out — v0.1 was 38.7; the v0.2 Thai sentiment set lifted it to {acc:.1f} (still the weakest Thai set)",
         "banking77": "whole dataset held out — weak on fine-grained 77-way intents (v0.1: 32.7)", "sib200_th": "whole dataset held out (v0.1: 77.5)"}
for label, src_key, qt in THAI:
    r = H[src_key][qt]; acc = 100 * r.get("acc", r.get("exact"))
    extra = f"MAE {r['mae']:.2f}" if qt == "score" else (f"F1 {r['macro_f1']:.3f}" if "macro_f1" in r else "")
    cell = f"**{acc:.1f}**" if acc >= 60 else f"{acc:.1f}"
    pat = rf"^\| {re.escape(label)} \| {qt} \| \d+ \| [^|]+\| [^|]*\| [^|]+\| (?P<note>[^|]+)\|$"
    mo = re.search(pat, s, flags=re.M); assert mo, (label, qt)
    note = NOTES.get(src_key, mo.group("note").strip()).format(acc=acc) if src_key in NOTES else mo.group("note").strip()
    s = s[:mo.start()] + f"| {label} | {qt} | {r['n']} | {cell} | {extra} | {r['ece']:.3f} | {note} |" + s[mo.end():]
s = s.replace("### Calibration (Stage 3)\n", "### Calibration (Stage 3)\n\nv0.2 learned temperatures: choice 1.055, noul 1.047, score 1.000 (the before/after table below was measured on v0.1;\nthe v0.2 procedure is identical).\n")
s = s.replace("""* v0.1 known weak spots (numbers above): Thai social-media sentiment (wisesight 38.7), fine-grained 77-way English
  intents (banking77 32.7), extractive-QA style yes/no (squad2 at chance), and summary *relevance* scoring. v0.2 adds a
  synthetic Thai sentiment set and a second SFT round; expect the first to move, not the others.""",
f"""* v0.2 known weak spots (numbers above): extractive-QA style yes/no (squad2 at chance, boolq {100*m(P['boolq'])[1]['acc']:.0f}, pubmedqa
  {100*m(P['pubmedqa'])[1]['acc']:.0f}), summary *relevance* scoring ({100*m(P['summeval-relevance'])[1]['exact']:.0f}), fine-grained 77-way English intents (banking77
  {100*H['banking77']['choice']['acc']:.0f}) and Thai social sentiment (wisesight {100*H['wisesight']['choice']['acc']:.0f}, up from 38.7). Targeted data for each is in progress for v0.3.""")
versions = f"""## Versions

| version | date | change | public macro | Wisesight |
|---|---|---|---|---|
| v0.2 | 2026-09-21 | +3,000 SFT steps from v0.1 with a 22k-record synthetic Thai social-sentiment set (4/3/5-class, yes/no, score schemes), re-calibrated | **{macro:.1f}** | {100*H['wisesight']['choice']['acc']:.1f} |
| v0.1 | 2026-09-20 | initial release: Thai CPT 4.47B tokens, 12k-step SFT, calibration | 61.9 | 38.7 |

"""
s = s.replace("## License & credits", versions + "## License & credits", 1)
open(out, "w", encoding="utf-8").write(s)
print(f"macro {macro:.1f}; card written -> {out}")
