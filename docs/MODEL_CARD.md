---
license: apache-2.0
language:
  - th
  - en
base_model: Qwen/Qwen3.5-0.8B-Base
pipeline_tag: zero-shot-classification
tags:
  - thai
  - zero-shot-classification
  - system-one
  - decision-model
  - computer-use
  - openthaigpt
  - iapp
---

# OpenThai-SystemOne

![OpenThai-SystemOne: Jev ภาษาไทยมาแล้ว — open Thai + English System One decision model](https://huggingface.co/iapp/OpenThai-SystemOne/resolve/main/assets/openthai-systemone-launch.jpg)

**An open Thai + English "System One" decision model.** It does not generate text. Given a *state* (any text or JSON)
and typed *questions*, it returns calibrated probabilities over the options in **one forward pass**:

| Question | You give | You get |
|---|---|---|
| `choice` | instructions + up to **255 options** (name → description or null) | `choice`, `probabilities`, `confidence` |
| `score`  | instructions + 2–10 ordered level descriptions | `score` (probability-weighted, fractional), `probabilities`, `confidence` |
| `noul`   | a yes/no question | `noul` = p(yes) |

The request/response contract mirrors TypeSafe's `POST /v1/systemone` so code written for the TypeSafe SDK can be
pointed at this model unchanged. Typical uses: ticket routing, moderation, intent detection, RAG relevance judging,
LLM-output verification, and **computer-use / browser-agent action selection** (which element to click, which tool to call).

## How it works

* Backbone: text tower of [Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) (24 layers, hybrid
  Gated-DeltaNet / attention, 262k context), vision encoder removed, then **continued-pretrained on ~5B tokens of Thai**
  (web, Wikipedia, parallel Thai↔English, and machine-state text such as accessibility trees and JSON).
* The 248k-token LM head is **replaced by a 256-way slot head**. Options are introduced by control tokens
  `<|ts_opt_0|> … <|ts_opt_254|>`; the hidden state at each `<|ts_answer|>` token is projected to 256 logits, slots beyond
  the number of options are masked, and a softmax gives the distribution. Slot 255 is *abstain* (none of the options fit).
* Trained on ~2–3M decision examples converted from public Thai/English classification, NLI, QA, rating, agent and
  tool-selection datasets plus synthetic Thai/English decision tasks, with option-order shuffling and abstain examples;
  then a short calibration stage (Brier loss + per-type temperature) so that higher confidence ⇒ higher accuracy.

## Usage

```bash
pip install openthai-systemone            # or: pip install "git+https://github.com/iapp-technology/openthai-systemone"
```

```python
from openthai_systemone import SystemOneClient, Choice, Score, Noul

client = SystemOneClient("iapp/OpenThai-SystemOne")
resp = client.system_one(
    state={"ticket": "ลูกค้าแจ้งว่าโดนหักเงินซ้ำสองครั้ง ขอเงินคืนด่วน โทรมาสามรอบแล้ว"},
    questions={
        "department": Choice(instructions="ทีมใดควรรับผิดชอบ", criteria={"billing": "การเงิน/ค่าบริการ", "technical": "ระบบใช้งานไม่ได้", "sales": None}),
        "frustration": Score(instructions="ลูกค้าหงุดหงิดแค่ไหน", criteria=["ใจเย็น", "หงุดหงิดแต่สุภาพ", "โกรธมาก"]),
        "refund_requested": Noul(instructions="ลูกค้าขอเงินคืนอย่างชัดเจนหรือไม่"),
    },
)
print(resp.answers["department"].choice, resp.answers["department"].probabilities)
print(resp.answers["frustration"].score, resp.answers["refund_requested"].noul)
```

HTTP server with the TypeSafe-compatible contract:

```bash
OPENTHAI_SYSTEMONE_MODEL=iapp/OpenThai-SystemOne uvicorn openthai_systemone.server:app --port 8000
curl -X POST localhost:8000/v1/systemone -H 'content-type: application/json' -d '{"state": "...", "questions": {...}}'
```

Or with plain transformers (remote code shipped in this repo):

```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("iapp/OpenThai-SystemOne", trust_remote_code=True)
```

## Evaluation

All numbers are zero-shot: the model sees only the state, the instructions and the option names/descriptions.

### Public benchmark (same 13 subsets, splits, instructions and sampler as Bespoke Nimble's `docs/PUBLIC_BENCHMARKS.md`)

Nimble-9B and Jev numbers are as published by Bespoke Labs (2026-09-18); ours are measured with `scripts/06_eval.py`
on `scripts/06b_public_benchmarks.py` rebuilds of the same subsets.

| subset | type | n | **OpenThai 0.8B** | Nimble-9B | Jev 1.13.0 | our ECE |
|---|---|---|---|---|---|---|
| aegis2 | noul | 250 | 58.0 | 81.2 | 80.4 | 0.243 |
| boolq | noul | 300 | 63.7 | 86.0 | 89.7 | 0.195 |
| civil_comments | noul | 300 | **78.0** | 70.3 | 81.0 | 0.127 |
| helpsteer2 | score | 250 | **42.8** | 39.0 | 34.1 | 0.400 |
| massive-de-DE | choice | 350 | 64.6 | 83.4 | 86.9 | 0.199 |
| massive-en-US | choice | 350 | 75.7 | 86.9 | 87.4 | 0.133 |
| multinli | choice | 299 | **85.6** | 85.3 | 82.9 | 0.035 |
| paws | noul | 250 | 67.2 | 82.8 | 89.2 | 0.143 |
| pubmedqa | choice | 250 | 53.6 | 75.6 | 77.2 | 0.154 |
| squad2 | noul | 299 | 50.2 | 80.6 | 82.9 | 0.461 |
| summeval-consistency | score | 144 | **84.0** | 75.7 | 81.2 | 0.061 |
| summeval-relevance | score | 240 | 13.8 | 49.2 | 35.0 | 0.791 |
| vitaminc-dev | choice | 599 | 67.1 | 76.6 | 80.1 | 0.098 |
| **macro average** | | | **61.9** | 74.8 | 76.0 | |

Reference: raw Qwen3.5-0.8B prompted with letter log-probs scores 45.4 macro on the same bench (Bespoke's number).
`choice`/`noul` report accuracy; `score` reports exact-level match. Bold = ahead of Bespoke-Nimble-9B.
Honest reading: the 0.8B model is ahead of the 9B on 4 of 13 subsets (NLI, summary consistency, helpfulness scoring,
toxicity) and clearly behind on reading-comprehension style yes/no tasks (squad2 is at chance, boolq, pubmedqa) and on
summary *relevance* scoring, which is the one subset where our score head is badly miscalibrated (ECE 0.79).

### Thai held-out sets (never in training; whole datasets held out where marked)

| set | type | n | accuracy | macro-F1 / MAE | ECE | note |
|---|---|---|---|---|---|---|
| MASSIVE-th intent (60-way) | choice | 5007 | **86.4** | F1 0.829 | 0.048 | eval split |
| Prachathai67k topics | choice | 3501 | **97.7** | F1 0.924 | 0.005 | eval split |
| Prachathai67k topics | noul | 13119 | **94.1** | | 0.008 | eval split |
| XNLI-th | choice | 2490 | **76.5** | F1 0.767 | 0.028 | eval split |
| XNLI-th (entailment yes/no) | noul | 2490 | **84.3** | | 0.042 | eval split |
| SIB-200 Thai topic (7-way) | choice | 204 | **77.5** | F1 0.752 | 0.074 | whole dataset held out |
| Thai contrastive pairs (one-fact flips) | choice | 296 | **78.7** | F1 0.724 | 0.105 | synthetic, eval-only |
| Thai contrastive pairs | score | 56 | **73.2** | MAE 0.48 | 0.133 | synthetic, eval-only |
| Thai contrastive pairs | noul | 248 | **82.3** | | 0.088 | synthetic, eval-only |
| Wongnai review stars (1–5) | score | 6203 | **63.3** | MAE 0.44 | 0.010 | eval split |
| Wisesight sentiment (4-class) | choice | 2671 | 38.7 | F1 0.357 | 0.341 | whole dataset held out — **known weak spot**, v0.2 targets it |
| banking77 intent (77-way, English) | choice | 3076 | 32.7 | F1 0.290 | 0.165 | whole dataset held out — weak on fine-grained 77-way intents |
| xLAM tool selection (English) | choice | 884 | **99.4** | F1 0.986 | 0.007 | eval slice |

Batch-1 latency, one question with 255 options, H100 shared with a training job: **44 ms** (public bench run),
48 ms (held-out run). A 3-question Thai ticket (166 tokens): ~40 ms on H100, 154 ms on a MacBook M3 Max (MPS).

### Calibration (Stage 3)

After SFT (12k steps) the backbone was frozen and the slot head plus one temperature per question type were trained for
400 steps on the SFT mixture with cross-entropy + Brier loss (`configs/calib.yaml`, `brier_weight: 1.0`,
`train_temperature: true`). Learned temperatures: choice 1.062, noul 1.047, score 1.008.

Before/after on the same sources (before = SFT checkpoint, 150–300-item smoke slices; after = calibrated checkpoint, full slices,
so accuracies are not strictly comparable, ECE is):

| set | type | ECE before → after | accuracy before → after |
|---|---|---|---|
| multinli | choice | 0.049 → **0.035** | 84.7 → 85.6 |
| massive-en-US | choice | 0.064 → 0.133 | 73.3 → 75.7 |
| boolq | noul | 0.086 → 0.195 | 69.3 → 63.7 |
| paws | noul | 0.108 → 0.143 | 69.3 → 67.2 |
| vitaminc-dev | choice | 0.104 → **0.098** | 58.0 → 67.1 |
| MASSIVE-th | choice | 0.032 → 0.048 | 89.7 → 86.4 |
| Prachathai (choice / noul) | | 0.033 / 0.024 → **0.005 / 0.008** | 97.4 / 93.4 → 97.7 / 94.1 |
| XNLI-th (choice / noul) | | 0.062 / 0.012 → **0.028 / 0.042** | 80.0 / 87.0 → 76.5 / 84.3 |
| SIB-200 th | choice | 0.062 → 0.074 | 77.5 → 77.5 |
| Wongnai stars | score | 0.071 → **0.010** | 66.3 → 63.3 |
| Thai contrastive (choice / noul) | | 0.083 / 0.075 → 0.105 / 0.088 | 79.1 / 84.6 → 78.7 / 82.3 |
| Wisesight | choice | 0.286 → 0.341 | 39.0 → 38.7 |

Reading: calibration helps where the model is already competent (Thai topic/NLI, Wongnai scoring, MultiNLI: ECE
≤ 0.05) and does not rescue sets where accuracy itself is low (wisesight, squad2, summeval-relevance) — a temperature
cannot fix a wrong ranking. On the English public bench the median ECE is 0.15; treat `confidence` as reliable on the
Thai sets and on NLI/topic tasks, and route low-confidence English yes/no decisions to a bigger model.

## Limits

* Text only. Up to 255 options per question in one stage (bucket into groups for more). 64k tokens per request.
* It is a small model: use the `confidence` field and route low-confidence cases to a bigger model or a human.
* Not a reasoning model: it will not do multi-step verification or arithmetic.
* v0.1 known weak spots (numbers above): Thai social-media sentiment (wisesight 38.7), fine-grained 77-way English
  intents (banking77 32.7), extractive-QA style yes/no (squad2 at chance), and summary *relevance* scoring. v0.2 adds a
  synthetic Thai sentiment set and a second SFT round; expect the first to move, not the others.
* English calibration is weaker than Thai (median ECE 0.15 vs ≤ 0.05): the training mix is Thai-heavy by design.

## License & credits

Apache-2.0. Built by iApp Technology / OpenThai on Qwen3.5-0.8B-Base (Apache-2.0). Inspired by TypeSafe AI's
System One models (Jev); this is an independent open re-implementation and is not affiliated with TypeSafe AI.
