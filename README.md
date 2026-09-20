# OpenThai-SystemOne

![OpenThai-SystemOne launch: Jev ภาษาไทยมาแล้ว — open Thai + English System One decision model](https://raw.githubusercontent.com/iapp-technology/openthai-systemone/main/docs/assets/fb_launch_v3_1x1.png)

Open-source Thai + English **System One decision model**: a Qwen3.5-0.8B text tower whose LM head is replaced by a
**256-way slot softmax**, continued-pretrained on Thai, and trained to answer typed questions (`choice` / `score` /
`noul`) about an arbitrary *state* in a single forward pass with calibrated probabilities. The request/response contract
mirrors TypeSafe AI's `POST /v1/systemone` (Jev), so SDK code written for TypeSafe can point at this model.

## Quickstart

```bash
pip install "git+https://github.com/iapp-technology/openthai-systemone"      # or: pip install openthai-systemone
```

```python
from openthai_systemone import SystemOneClient, Choice, Score, Noul

client = SystemOneClient("iapp/OpenThai-SystemOne")          # CUDA, MPS or CPU
resp = client.system_one(
    state={"ticket": "โดนหักเงินซ้ำสองครั้ง ขอเงินคืนด่วน โทรไปสามรอบแล้วไม่มีใครรับ"},
    questions={
        "department": Choice(instructions="ทีมใดควรรับผิดชอบ", criteria={"billing": "การเงิน/คืนเงิน", "technical": "ระบบใช้งานไม่ได้", "sales": None}),
        "frustration": Score(instructions="ลูกค้าหงุดหงิดแค่ไหน", criteria=["ใจเย็น", "หงุดหงิดแต่สุภาพ", "โกรธมาก"]),
        "refund": Noul(instructions="ลูกค้าขอเงินคืนอย่างชัดเจนหรือไม่"),
    },
)
print(resp.answers["department"].choice, resp.answers["department"].probabilities)
print(resp.answers["frustration"].score, resp.answers["refund"].noul)
```

HTTP server with the same JSON contract as TypeSafe's `POST /v1/systemone`:

```bash
OPENTHAI_SYSTEMONE_MODEL=iapp/OpenThai-SystemOne uvicorn openthai_systemone.server:app --port 8000
```

Model card with benchmark tables: https://huggingface.co/iapp/OpenThai-SystemOne.

## Layout

```
openthai_systemone/     pip package: types (Choice/Score/Noul), formatting (prompt + slots), modeling (SlotHead), client, server
scripts/00..07_*.py    one script per pipeline stage (see below)
configs/*.yaml         CPT / SFT / calibration hyper-parameters (1x H100)
tests/                 CPU unit tests
```

## Pipeline

| stage | script | where | output |
|---|---|---|---|
| 0 surgery | `scripts/00_prepare_base.py` | CPU | `base/qwen3.5-0.8b-text` (vision encoder dropped) |
| 1 CPT data | `scripts/01_cpt_data.py` | CPU, needs internet | `data/cpt/shard-*.bin` (5B Thai-heavy tokens) |
| 1 CPT | `scripts/02_cpt_train.py --config configs/cpt.yaml` | 1x H100, ~1 day | `runs/cpt/latest` |
| 2 decision data | `scripts/03_decision_data.py`, `scripts/03b_synth_generate.py` | CPU + LLM endpoint | `data/decision/*.jsonl`, `data/synth/*.jsonl` |
| 2 SFT | `scripts/04_decision_train.py --config configs/sft.yaml` | 1x H100, ~0.5 day | `runs/sft/latest` |
| 3 calibration | `scripts/04_decision_train.py --config configs/calib.yaml --init runs/sft/latest` | 1x H100, hours | `runs/calib/latest` |
| 4 eval | `scripts/06_eval.py --model runs/calib/latest` | GPU | `runs/eval.json` |
| 5 release | `scripts/07_export_release.py --ckpt runs/calib/latest --out release/... --push iapp/OpenThai-SystemOne` | – | HF Hub |

## Dev

```bash
uv venv -p 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[server,train,dev]"
pytest -q
```

Synthetic data needs an OpenAI-compatible endpoint:

```bash
export SYNTH_BASE_URL=http://localhost:8000/v1 SYNTH_MODEL=<any-openai-compatible-chat-model> SYNTH_API_KEY=<key-or-none>
python scripts/03b_synth_generate.py --out data/synth --n 600000 --concurrency 32
```
