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

![OpenThai-SystemOne: an open Thai + English System One decision model, 0.8B, Apache-2.0](https://huggingface.co/iapp/OpenThai-SystemOne/resolve/main/assets/openthai-systemone-launch-en.png)

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

## Demo: playing Doom, no vision, no text

![OpenThai-SystemOne playing Doom](https://huggingface.co/iapp/OpenThai-SystemOne/resolve/main/assets/doom_10s.gif)

The model controls a Doom marine ([ViZDoom](https://github.com/Farama-Foundation/ViZDoom)) in real time. Every 4 game
tics the engine's symbolic state is serialised to text, for example:

```text
health 100/100 | ammo 50 | kills 1
crosshair: empty; nearest visible enemy 39deg to the RIGHT
enemies: Zombieman 5m right -39deg VISIBLE; ChaingunGuy 19m right -24deg; Zombieman 19m ahead -12deg
items: GreenArmor 41m right -18deg
depth ahead: 35/255 (obstacle near)
last actions: ATTACK ATTACK ATTACK ATTACK
```

and the model answers two typed questions in one forward pass: a `choice` over the 7 actions (the key that gets
pressed) and a `noul` "is an enemy in the crosshair". About 41 ms per decision on one H100 (~24 decisions/s),
0 output tokens, and no Doom data in training: everything comes from reading the state and the option descriptions.
It misses shots and dies on hard levels; the point is the speed and the calibrated probabilities, the same mechanics
that route tickets or pick UI elements for an agent.

Run it on your machine (iApp API key or the local weights, live HUD in Thai or English, optional recording):
**https://github.com/iapp-technology/openthai-systemone-doom**

## Thai showcase (real v0.3 outputs, 2026-09-22, via the API)

Seven everyday Thai tasks, each answered in one forward pass; probabilities are the model's actual output.

**1. Support ticket triage** — state: *"แอปโอนเงินไม่ได้ตั้งแต่เมื่อคืน ขึ้นว่า error 502 ตลอด ลองลงใหม่แล้วก็ยังไม่หาย รบกวนช่วยด่วนนะครับ ต้องโอนค่าเทอมลูกพรุ่งนี้"*

| question | type | answer |
|---|---|---|
| ทีมใดควรรับผิดชอบ (billing / technical / sales / account) | choice | **technical 72%**, billing 27% |
| ความเร่งด่วน (ไม่เร่งด่วน → วิกฤต) | score | **2.13 = เร่งด่วน** (85%), วิกฤต 14% |
| ลูกค้าใช้ถ้อยคำสุภาพหรือไม่ | noul | 1% (the ticket is polite... see note) |

**2. News topic** — *"ครม. เห็นชอบขึ้นค่าแรงขั้นต่ำเป็น 400 บาททั่วประเทศ มีผล 1 มกราคม สภาอุตสาหกรรมกังวลกระทบ SME"* (8 topics)
→ choice **เศรษฐกิจ 62%**, แรงงาน 36%; noul "เกี่ยวกับแรงงานหรือไม่" → **98%**. Both readings are right; the probabilities show the overlap instead of hiding it.

**3. Assistant intent** — *"ช่วยตั้งปลุกตอนหกโมงครึ่งพรุ่งนี้ให้หน่อย แล้วก็เปิดเพลงเบาๆ ตอนตื่นด้วย"* (10 intents)
→ **alarm set 99.5%** (the secondary "play music" request does not distract the primary intent).

**4. Comment moderation** — *"ไอ้พวกเหี้ย ทำงานกันแบบนี้ไปตายซะ ใครก็ได้เอาคนพวกนี้ออกไปที"*
→ noul เป็นพิษ **94%**; choice sentiment **เชิงลบ 97%**.

**5. Grounded QA (answerable or not)** — passage about สะพานพระราม 8 (opened 7 May 2545, 475 m long, Bangphlat ↔ Phra Nakhon)
→ "สะพานยาวเท่าไร" answerable **98.5%**; "ระบุงบประมาณก่อสร้างหรือไม่" **0.6%**. It says *yes* only when the fact is actually in the text.

**6. Agent tool selection** — task "จองโต๊ะร้านอาหาร 4 คน คืนนี้สองทุ่ม", 5 tools, history shows `search_restaurants` already found a free table
→ next tool **make_reservation 96.5%** (not search again, not SMS yet).

**7. RAG relevance + extraction** — question on withholding tax for building rent, passage stating 5%
→ passage relevant **98%**; rate choice **5% (76%)**, 3% 16%.

Latency from a laptop over the internet was 170–220 ms per request including network; the model itself takes ~40 ms on an H100.
Note on example 1's politeness flag: the ticket uses "รบกวน…นะครับ", so the correct answer is *yes*; the model said no
at 99%. Politeness judgement on formal Thai is a known gap and will get a targeted set in v0.4. We keep the miss here
because a showcase that only shows hits is not useful.

## Evaluation

All numbers are zero-shot: the model sees only the state, the instructions and the option names/descriptions.

### Public benchmark (same 13 subsets, splits, instructions and sampler as Bespoke Nimble's `docs/PUBLIC_BENCHMARKS.md`)

Nimble-9B and Jev numbers are as published by Bespoke Labs (2026-09-18); ours are measured with `scripts/06_eval.py`
on `scripts/06b_public_benchmarks.py` rebuilds of the same subsets.

| subset | type | n | **OpenThai 0.8B** | Nimble-9B | Jev 1.13.0 | our ECE |
|---|---|---|---|---|---|---|
| aegis2 | noul | 250 | **83.2** | 81.2 | 80.4 | 0.065 |
| boolq | noul | 300 | 79.7 | 86.0 | 89.7 | 0.049 |
| civil_comments | noul | 300 | **79.0** | 70.3 | 81.0 | 0.087 |
| helpsteer2 | score | 250 | **41.6** | 39.0 | 34.1 | 0.371 |
| massive-de-DE | choice | 350 | **88.3** | 83.4 | 86.9 | 0.057 |
| massive-en-US | choice | 350 | **88.3** | 86.9 | 87.4 | 0.070 |
| multinli | choice | 299 | **89.0** | 85.3 | 82.9 | 0.053 |
| paws | noul | 250 | **94.0** | 82.8 | 89.2 | 0.035 |
| pubmedqa | choice | 250 | 64.0 | 75.6 | 77.2 | 0.259 |
| squad2 | noul | 299 | **89.3** | 80.6 | 82.9 | 0.041 |
| summeval-consistency | score | 144 | 75.0 | 75.7 | 81.2 | 0.076 |
| summeval-relevance | score | 240 | 21.7 | 49.2 | 35.0 | 0.356 |
| vitaminc-dev | choice | 599 | 72.5 | 76.6 | 80.1 | 0.118 |
| **macro average** | | | **74.3** | 74.8 | 76.0 | |

For scale: Bespoke reports raw Qwen3.5-0.8B at 45.4 on their *private* 324-item holdout (not this bench), Nimble-9B at 90.1, Jev at 93.2.
`choice`/`noul` report accuracy; `score` reports exact-level match. Bold = ahead of Bespoke-Nimble-9B.
Honest reading: the 0.8B model is ahead of the 9B on 4 of 13 subsets (NLI, summary consistency, helpfulness scoring,
toxicity) and clearly behind on reading-comprehension style yes/no tasks (squad2 is at chance, boolq, pubmedqa) and on
summary *relevance* scoring, which is the one subset where our score head is badly miscalibrated (ECE 0.79).

### Thai held-out sets (never in training; whole datasets held out where marked)

| set | type | n | accuracy | macro-F1 / MAE | ECE | note |
|---|---|---|---|---|---|---|
| MASSIVE-th intent (60-way) | choice | 5007 | **90.0** | F1 0.869 | 0.043 | eval split |
| Prachathai67k topics | choice | 3501 | **98.1** | F1 0.938 | 0.004 | eval split |
| Prachathai67k topics | noul | 13119 | **94.2** |  | 0.008 | eval split |
| XNLI-th | choice | 2490 | **77.1** | F1 0.772 | 0.045 | eval split |
| XNLI-th (entailment yes/no) | noul | 2490 | **84.3** |  | 0.049 | eval split |
| SIB-200 Thai topic (7-way) | choice | 204 | **77.9** | F1 0.759 | 0.084 | whole dataset held out (v0.1: 77.5) |
| Thai contrastive pairs (one-fact flips) | choice | 296 | **80.7** | F1 0.734 | 0.098 | synthetic, eval-only |
| Thai contrastive pairs | score | 56 | **78.6** | MAE 0.35 | 0.156 | synthetic, eval-only |
| Thai contrastive pairs | noul | 248 | **83.5** |  | 0.109 | synthetic, eval-only |
| Wongnai review stars (1–5) | score | 6203 | **63.5** | MAE 0.44 | 0.039 | eval split |
| Wisesight sentiment (4-class) | choice | 2671 | 51.6 | F1 0.448 | 0.353 | whole dataset held out — v0.1 38.7 → v0.2 51.5 → now 51.6 (weakest Thai set; use order-invariant mode) |
| banking77 intent (77-way, English) | choice | 3076 | 45.4 | F1 0.417 | 0.236 | whole dataset held out — 77-way near-duplicate intents (v0.1 32.7; 61.7 with order-invariant mode on v0.2) |
| xLAM tool selection (English) | choice | 884 | **99.4** | F1 0.986 | 0.006 | eval slice |

Batch-1 latency, one question with 255 options, H100 shared with a training job: **44 ms** (public bench run),
48 ms (held-out run). A 3-question Thai ticket (166 tokens): ~40 ms on H100, 154 ms on a MacBook M3 Max (MPS).

### Calibration (Stage 3)

v0.3 learned temperatures: choice 1.055, noul 1.047, score 1.008 (the before/after table below was measured on v0.1;
the procedure is identical in every version).

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
* v0.3 known weak spots (numbers above): summary *relevance* scoring (22) and helpfulness
  scoring (42) — the two 5-level rating tasks; fine-grained 77-way English intents (banking77 45 single-order); Thai social
  sentiment (wisesight 52); PubMedQA 3-way (64).
* English calibration is weaker than Thai (median ECE 0.15 vs ≤ 0.05): the training mix is Thai-heavy by design.

## Versions

| version | date | change | public macro | Wisesight |
|---|---|---|---|---|
| v0.3 | 2026-09-22 | +5,000 SFT steps from v0.2 with 177k real train-split records + 78k targeted synthetic records for the weak spots (grounded QA, summary rating, fine-grained intents, safety, paraphrase), re-calibrated | **74.3** | 51.6 |
| v0.2 | 2026-09-21 | +3,000 SFT steps from v0.1 with a 22k-record synthetic Thai social-sentiment set (4/3/5-class, yes/no, score schemes), re-calibrated | **63.2** | 51.5 |
| v0.1 | 2026-09-20 | initial release: Thai CPT 4.47B tokens, 12k-step SFT, calibration | 61.9 | 38.7 |

## Changelog

**v0.3 — 2026-09-22**
- Continued fine-tuning for 5,000 steps from v0.2 with the weak-spot data: real train splits (SQuAD2, BoolQ, PubMedQA-artificial, PAWS, Aegis2, ToxicChat, XQuAD-th, MASSIVE-de; 177k records) and five targeted synthetic sets generated with Qwen3.6-35B-A3B and blind-checked (grounded yes/no QA with near-miss unanswerables 30k, summary rating on the SummEval rubrics 20k, 40–120-way near-duplicate intent taxonomies 12k, safety judgments 8k, adversarial paraphrases 8k); sentiment set weight lowered from 4× to 2×; re-calibrated.
- Public 13-subset macro 63.2 → **74.3** (Nimble-9B 74.8): SQuAD2 50.2 → 89.3, PAWS 68.0 → 94.0, Aegis2 61.6 → 83.2, MASSIVE-de 67.4 → 88.3, BoolQ 64.7 → 79.7, MASSIVE-en 79.1 → 88.3, PubMedQA 56.4 → 64.0, SummEval-relevance 14.2 → 21.7, VitaminC 68.6 → 72.5, MultiNLI 87.3 → 89.0. ECE improved on 11 of 13 subsets.
- Regression: SummEval-consistency 84.0 → 75.0 (the synthetic consistency set only covers levels 1/3/5; levels 2/4 will be added). Thai held-out sets all flat or up (SIB-200 74.0 → 77.9, banking77 37.8 → 45.4 single-order).
- Note: SQuAD2, BoolQ, PAWS, Aegis2 and MASSIVE-de are no longer "never trained on": their *train* splits are now in the mix; the public-bench subsets still use the validation/test splits only.

**v0.2 — 2026-09-21**
- Continued fine-tuning for 3,000 steps from v0.1 with a new 22k-record synthetic Thai social-media sentiment set
  (4-class with the "question" class, 3/5-class variants, yes/no flags, 5-level score), then re-calibrated.
- Public 13-subset macro 61.9 → **63.2**; Wisesight 38.7 → **51.5**; MASSIVE-en 75.7 → 79.1; MASSIVE-th 86.4 → 88.6;
  MultiNLI 85.6 → 87.3 (ECE 0.035 → 0.016); Aegis2 58.0 → 61.6; PubMedQA 53.6 → 56.4.
- Regression: SIB-200 Thai 77.5 → 74.0 (sentiment set was weighted 4×; will be lowered next round). Unchanged: SQuAD2, SummEval-relevance.
- Known issue and fix: like Jev, a single forward pass is sensitive to the *order* of the options (measured on v0.2:
  5–12% of arg-max choices flip under a random permutation for ≤ 10 options, 72% for 77-way banking77). The client and
  API now have an **order-invariant mode** that averages the answer over several cyclic option orders in one batched
  pass (`order_invariant: true` / `permutations: n`; automatic for choice questions with > 10 options). With it, on 360
  held-out Thai questions: flips 18.9% → 3.3%, mean probability shift 0.19 → 0.08, accuracy 68.6 → 73.6, banking77
  41.7 → 63.3. Cost: ~2× latency (batched), never more tokens per request in `usage.input_tokens`.
- Live API switched to v0.2 (same checkpoint as these weights).

**v0.1 — 2026-09-20**
- Initial release: Qwen3.5-0.8B text tower, 4.47B-token Thai CPT, 256-slot decision head, 12k-step SFT on 1.8M
  public + 98k synthetic decision records, calibration stage. Public macro 61.9.

## License & credits

Apache-2.0. Built by iApp Technology / OpenThai on Qwen3.5-0.8B-Base (Apache-2.0). Inspired by TypeSafe AI's
System One models (Jev); this is an independent open re-implementation and is not affiliated with TypeSafe AI.

## Sponsor

<a href="https://siam.ai" target="_blank" rel="noopener"><img src="https://huggingface.co/iapp/OpenThai-SystemOne/resolve/main/assets/siamai-logo.png" alt="Siam AI Corporation" width="180" /></a>

Training, evaluation and the free hosted API for this model run on NVIDIA H100 GPUs generously provided by [Siam AI Corporation](https://siam.ai). Thank you for backing open Thai AI.
