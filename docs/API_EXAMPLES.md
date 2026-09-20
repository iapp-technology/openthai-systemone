# OpenThai-SystemOne — API call examples (curl)

Endpoint contract: `POST /v1/systemone`, JSON in, JSON out, no text generation (`output_tokens` is always 0).
Same shape as TypeSafe's Jev API: one `state` (string or any JSON) + a map of typed `questions`.

| Where | URL | Auth | Who can call |
|---|---|---|---|
| **Public (iApp API Gateway)** | `https://api.iapp.co.th/v3/store/openthai/systemone` | header `apikey: iapp_live_...` | anyone with an iApp API key |
| Local | `http://localhost:8000/v1/systemone` after `OPENTHAI_SYSTEMONE_MODEL=iapp/OpenThai-SystemOne uvicorn openthai_systemone.server:app --port 8000` | none | you |

Set once:
```bash
export IAPP_API_KEY="iapp_live_xxxxxxxxxxxxxxxx"
export URL="https://api.iapp.co.th/v3/store/openthai/systemone"
```

## 1. Thai support ticket → department + frustration level + refund? (all three in one call)

```bash
curl -s "$URL" -H "apikey: $IAPP_API_KEY" -H "content-type: application/json" -d '{
  "state": {"ticket": "โดนหักเงินซ้ำสองครั้งเมื่อวานนี้ ขอเงินคืนด่วนนะครับ โทรไปสามรอบแล้วไม่มีใครรับ"},
  "questions": {
    "department":  {"type": "choice", "instructions": "ทีมใดควรรับผิดชอบ",
                    "criteria": {"billing": "การเงิน/คืนเงิน", "technical": "ระบบใช้งานไม่ได้", "sales": null}},
    "frustration": {"type": "score",  "instructions": "ลูกค้าหงุดหงิดแค่ไหน",
                    "criteria": ["ใจเย็น", "หงุดหงิดแต่สุภาพ", "โกรธมาก"]},
    "refund":      {"type": "noul",   "instructions": "ลูกค้าขอเงินคืนอย่างชัดเจนหรือไม่"}
  }
}'
```
Real response (2026-09-20, SFT checkpoint):
```json
{"model":"openthai-systemone",
 "answers":{
   "department":{"type":"choice","choice":"billing","probabilities":{"billing":0.963,"technical":0.011,"sales":0.026},"confidence":0.83,"abstain":0.085},
   "frustration":{"type":"score","score":1.95,"legend":{"0":"ใจเย็น","1":"หงุดหงิดแต่สุภาพ","2":"โกรธมาก"},"probabilities":{"0":0.012,"1":0.031,"2":0.957},"confidence":0.82},
   "refund":{"type":"noul","noul":0.946}},
 "usage":{"input_tokens":166,"output_tokens":0}}
```

## 2. Comment moderation (Thai): toxic? + sentiment

```bash
curl -s "$URL" -H "apikey: $IAPP_API_KEY" -H "content-type: application/json" -d '{
  "state": "แอปห่วยมาก โอนเงินไม่ผ่านสามวันแล้ว ใครก็ได้ช่วยตอบที",
  "questions": {
    "toxic":     {"type": "noul", "instructions": "ข้อความนี้เป็นข้อความที่เป็นพิษ (ด่าทอ เหยียด คุกคาม) หรือไม่"},
    "sentiment": {"type": "choice", "instructions": "ข้อความนี้แสดงความรู้สึกแบบใด",
                  "criteria": {"เชิงบวก": null, "เป็นกลาง": null, "เชิงลบ": null, "คำถาม": "ถามหาข้อมูลหรือความเห็น"}}
  }
}'
```

## 3. Computer-use / browser agent: which element to act on, and how

```bash
curl -s "$URL" -H "apikey: $IAPP_API_KEY" -H "content-type: application/json" -d '{
  "state": {
    "task": "กรอกอีเมลแล้วกดชำระเงิน",
    "screen": "[3] heading \"ชำระเงิน\"\n[7] textbox \"อีเมล\" value=\"\"\n[8] textbox \"เบอร์โทร\" value=\"0812345678\"\n[9] checkbox \"ยอมรับเงื่อนไข\" checked=true\n[12] button \"ชำระเงิน\"\n[13] link \"ยกเลิก\""
  },
  "questions": {
    "target":    {"type": "choice", "instructions": "Which element should the agent act on next?",
                  "criteria": {"[7] textbox อีเมล": null, "[8] textbox เบอร์โทร": null, "[9] checkbox ยอมรับเงื่อนไข": null, "[12] button ชำระเงิน": null, "[13] link ยกเลิก": null}},
    "operation": {"type": "choice", "instructions": "Which operation should be performed on it?",
                  "criteria": {"CLICK": null, "TYPE": null, "SELECT": null}}
  }
}'
```

## 4. RAG relevance judge (English)

```bash
curl -s "$URL" -H "apikey: $IAPP_API_KEY" -H "content-type: application/json" -d '{
  "state": {"question": "When was the Thai baht floated?",
            "passage": "On 2 July 1997 the Bank of Thailand abandoned the peg to the US dollar and let the baht float."},
  "questions": {"answers": {"type": "noul", "instructions": "Does the passage answer the question? Use only what the passage states."}}
}'
```

## 5. Many options at once (up to 255) — product category with 40 choices

```bash
python3 - <<'EOF' | curl -s "$URL" -H "apikey: $IAPP_API_KEY" -H "content-type: application/json" -d @-
import json
cats = ["เสื้อผ้าผู้ชาย","เสื้อผ้าผู้หญิง","รองเท้า","กระเป๋า","นาฬิกา","มือถือ","แท็บเล็ต","โน้ตบุ๊ก","หูฟัง","ลำโพง",
        "ทีวี","ตู้เย็น","เครื่องซักผ้า","แอร์","พัดลม","เครื่องครัว","ของใช้ในบ้าน","เฟอร์นิเจอร์","ที่นอน","โคมไฟ",
        "สกินแคร์","เครื่องสำอาง","น้ำหอม","อาหารเสริม","ยา","ของเล่น","ของใช้เด็ก","สัตว์เลี้ยง","กีฬา","จักรยาน",
        "รถยนต์","มอเตอร์ไซค์","หนังสือ","เครื่องเขียน","เกม","กล้อง","เครื่องดนตรี","ต้นไม้","อาหารแห้ง","เครื่องดื่ม"]
print(json.dumps({"state": "เสื้อยืดคอกลมผ้าฝ้าย 100% สีขาว ไซส์ L ราคา 299 บาท",
                  "questions": {"category": {"type": "choice", "instructions": "สินค้านี้อยู่ในหมวดใด", "criteria": {c: None for c in cats}}}},
                 ensure_ascii=False))
EOF
```

## 6. Health check

```bash
curl -s https://api.iapp.co.th/v3/store/openthai/systemone/healthz -H "apikey: $IAPP_API_KEY"      # via gateway (if the route forwards GET)
```

## Reading the response

- `choice` → `choice` (best option), `probabilities` (sum to 1 over your options), `confidence` (1 − normalised entropy), `abstain` (P(none fits); our extension).
- `score` → `score` = Σ p_i·i (fractional), `probabilities` per level, `legend`.
- `noul` → `noul` = P(yes).
- Route low-`confidence` answers to a bigger model or a human. Max 255 options per `choice`, 2–10 levels per `score`, 64k tokens per request.

## Errors

| status | meaning |
|---|---|
| 401 | missing/invalid `apikey` (gateway) |
| 403 | you are calling the origin directly; use `api.iapp.co.th` |
| 422 | malformed request (e.g. `score` with fewer than 2 levels, `choice` with > 255 options) |
| 429 | rate limit (gateway per-key; origin 20 req/s per source) |
