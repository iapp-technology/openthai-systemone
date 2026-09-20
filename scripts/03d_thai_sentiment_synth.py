"""Thai social-media sentiment synthetic data (fix for the wisesight-style weak spot).

    export SYNTH_BASE_URL=http://localhost:8000/v1 SYNTH_MODEL=<any-openai-compatible-chat-model> SYNTH_API_KEY=none
    python scripts/03d_thai_sentiment_synth.py --out data/synth_sentiment --n 20000 --concurrency 24

Each call generates a batch of short, realistic Thai social-media texts (tweets, FB/Pantip comments, LINE messages,
reviews) with a gold class from {positive, neutral, negative, question}; every text is blind re-classified in a
second call and kept only when the labels agree. Each kept text is emitted under a randomly chosen question scheme:
  4-class (wisesight-style, Thai or English option names, with/without descriptions), 3-class (no "question"),
  5-class (+ mixed), noul ("is this a question?", "is this negative?", ...), score (5 sentiment levels).
Records: source="synth_sentiment_th", lang="th". wisesight itself is NEVER used (it is a held-out eval set).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
gen = __import__("03b_synth_generate")
from openthai_systemone.data.records import Record  # noqa: E402

CLASSES = ["positive", "neutral", "negative", "question"]
TH = {"positive": "เชิงบวก", "neutral": "เป็นกลาง", "negative": "เชิงลบ", "question": "คำถาม", "mixed": "ผสม"}
DESC_TH = {"positive": "พอใจ ชม ดีใจ แนะนำ", "neutral": "บอกเล่า ข้อมูล ไม่แสดงอารมณ์ชัด", "negative": "ไม่พอใจ บ่น ด่า ผิดหวัง",
           "question": "ถามหาข้อมูลหรือความเห็น ต้องการคำตอบ", "mixed": "มีทั้งชอบและไม่ชอบในข้อความเดียว"}
DESC_EN = {"positive": "satisfied, praising, happy, recommending", "neutral": "informational, no clear emotion",
           "negative": "dissatisfied, complaining, angry, disappointed", "question": "asking for information or opinions, expects an answer",
           "mixed": "both positive and negative in one message"}

DOMAINS = ["โทรศัพท์มือถือ/ค่ายมือถือ", "ธนาคาร/แอปธนาคาร", "ร้านอาหาร/เดลิเวอรี่", "เครื่องสำอาง/สกินแคร์", "ช้อปปิ้งออนไลน์/ขนส่งพัสดุ",
           "รถไฟฟ้า/ขนส่งสาธารณะ", "เกม/แกดเจ็ต", "ซีรีส์/ดารา/คอนเสิร์ต", "โรงพยาบาล/ประกัน", "ท่องเที่ยว/โรงแรม/สายการบิน",
           "มหาวิทยาลัย/การเรียน", "งาน/บริษัท/HR", "รถยนต์/ปั๊มน้ำมัน", "อินเทอร์เน็ตบ้าน/ไฟฟ้า/ประปา", "กีฬา/ฟุตบอล", "การเมืองแบบเบาๆ/ข่าวทั่วไป"]
PLATFORMS = ["ทวิตเตอร์/X", "คอมเมนต์เฟซบุ๊ก", "กระทู้ Pantip", "แชท LINE OA กับร้านค้า", "รีวิวใน Shopee/Lazada", "คอมเมนต์ TikTok", "รีวิว Google Maps"]

GEN_TEMPLATE = """สร้างข้อความโซเชียลมีเดียภาษาไทยที่สมจริง {k} ข้อความ หัวข้อ: {domain} แพลตฟอร์ม: {platform}
ต้องกระจายคลาสให้ครบ: positive {n_pos}, neutral {n_neu}, negative {n_neg}, question {n_q}
ลักษณะ: สั้น 5-45 คำ ภาษาพูด มีคำสแลง อีโมจิ สะกดผิดบ้าง ปนอังกฤษบ้าง แฮชแท็กบ้าง บางข้อความกำกวมเล็กน้อยแต่ตัดสินได้
"question" = ถามหาข้อมูล/ความเห็นจริงๆ (ไม่ใช่คำถามเชิงบ่นแบบ "ทำไมช้าจัง!!" ซึ่งเป็น negative)
"neutral" = บอกเล่าข้อมูล ประกาศ อัปเดตสถานะ ไม่แสดงอารมณ์ชัด
ตอบเป็น JSON เท่านั้น: {{"items": [{{"text": "...", "label": "positive|neutral|negative|question"}}, ...]}}"""

LABEL_TEMPLATE = """จำแนกข้อความโซเชียลภาษาไทยแต่ละข้อความเป็นคลาสเดียวจาก positive / neutral / negative / question
(question = ถามหาข้อมูลหรือความเห็นจริงๆ; คำถามเชิงบ่นให้ถือเป็น negative; neutral = บอกเล่า ไม่แสดงอารมณ์)
ตอบเป็น JSON เท่านั้น: {{"labels": ["...", ...]}} เรียงตามลำดับข้อความ

{items}"""

Q_TEMPLATES_TH = ["ข้อความนี้แสดงความรู้สึกแบบใด", "จัดประเภทอารมณ์ของโพสต์นี้", "ผู้เขียนรู้สึกอย่างไร", "โพสต์นี้เป็นความเห็นแบบไหน"]
Q_TEMPLATES_EN = ["What is the sentiment of this message?", "Classify the sentiment of this social post.", "How does the writer feel?"]


def make_records(text: str, label: str, platform: str, domain: str, rng: random.Random, idx: str):
    """Emit 1-2 records for one labelled text under random schemes."""
    out = []
    schemes = rng.sample(["4th", "4en", "3", "5", "noul", "score"], k=rng.choice([1, 1, 2]))
    state = text if rng.random() < 0.6 else {"platform": platform, "topic": domain, "text": text}
    for s in schemes:
        use_desc = rng.random() < 0.5
        if s in ("4th", "4en", "3", "5"):
            classes = list(CLASSES) if s in ("4th", "4en") else (["positive", "neutral", "negative"] if s == "3" else CLASSES + ["mixed"])
            if s == "3" and label == "question":
                continue  # question texts don't fit a 3-class scheme
            thai = s == "4th" or (s in ("3", "5") and rng.random() < 0.6)
            names = {c: (TH[c] if thai else c) for c in classes}
            descs = {names[c]: ((DESC_TH if thai else DESC_EN)[c] if use_desc else None) for c in classes}
            rng.shuffle(classes)
            q = {"type": "choice", "instructions": rng.choice(Q_TEMPLATES_TH if thai else Q_TEMPLATES_EN), "criteria": {names[c]: descs[names[c]] for c in classes}}
            out.append(Record(f"{idx}-{s}", "synth_sentiment_th", "th", state, {"sentiment": q}, {"sentiment": names[label]}))
        elif s == "noul":
            target, inst, td, fd = rng.choice([
                ("question", "ข้อความนี้เป็นคำถามที่ต้องการคำตอบหรือไม่", "ถามหาข้อมูลหรือความเห็น", "ไม่ได้ถาม หรือเป็นแค่คำถามเชิงบ่น"),
                ("negative", "ข้อความนี้แสดงความไม่พอใจหรือไม่", "บ่น ด่า ผิดหวัง", "ไม่ได้แสดงความไม่พอใจ"),
                ("positive", "ผู้เขียนพอใจหรือไม่", "ชม ดีใจ แนะนำ", "ไม่ได้แสดงความพอใจ"),
                ("neutral", "Is this message emotionally neutral?", "informational, no clear emotion", "shows a clear positive/negative feeling or asks a question"),
            ])
            q = {"type": "noul", "instructions": inst}
            if use_desc:
                q["criteria"] = {"true": td, "false": fd}
            out.append(Record(f"{idx}-noul", "synth_sentiment_th", "th", state, {"flag": q}, {"flag": label == target}))
        elif s == "score":
            if label == "question":
                continue
            levels = ["แย่มาก/โกรธ", "ไม่พอใจ", "เฉยๆ/บอกเล่า", "พอใจ", "ประทับใจมาก"]
            lvl = {"negative": rng.choice([0, 1]), "neutral": 2, "positive": rng.choice([3, 4])}[label]
            q = {"type": "score", "instructions": rng.choice(["ระดับความพึงพอใจของผู้เขียน", "Rate the writer's sentiment from very negative to very positive"]), "criteria": levels}
            out.append(Record(f"{idx}-score", "synth_sentiment_th", "th", state, {"level": q}, {"level": lvl}))
    return out


async def one_batch(client, model, rng, sem, k=10):
    async with sem:
        domain, platform = rng.choice(DOMAINS), rng.choice(PLATFORMS)
        counts = [k // 4] * 4
        for i in rng.sample(range(4), k % 4):
            counts[i] += 1
        prompt = GEN_TEMPLATE.format(k=k, domain=domain, platform=platform, n_pos=counts[0], n_neu=counts[1], n_neg=counts[2], n_q=counts[3])
        try:
            r = await client.chat.completions.create(model=model, temperature=1.0, max_tokens=2500,
                                                     messages=[{"role": "system", "content": gen.SYSTEM}, {"role": "user", "content": prompt}],
                                                     extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            d = gen.parse_json(r.choices[0].message.content or "") or {}
            items = [it for it in d.get("items", []) if isinstance(it, dict) and isinstance(it.get("text"), str) and it.get("label") in CLASSES and it["text"].strip()]
            items = [it for it in items if not gen.has_bad_script(it["text"])]
            if not items:
                return [], {"invalid": 1}
            listing = "\n".join(f"{i+1}. {it['text']}" for i, it in enumerate(items))
            r2 = await client.chat.completions.create(model=model, temperature=0.0, max_tokens=400,
                                                      messages=[{"role": "system", "content": gen.SYSTEM}, {"role": "user", "content": LABEL_TEMPLATE.format(items=listing)}],
                                                      extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            labs = (gen.parse_json(r2.choices[0].message.content or "") or {}).get("labels", [])
            kept, stats = [], {"ok": 0, "disagree": 0}
            for it, lab in zip(items, labs):
                if isinstance(lab, str) and lab.strip().lower() == it["label"]:
                    kept.append((it["text"].strip(), it["label"], platform, domain))
                    stats["ok"] += 1
                else:
                    stats["disagree"] += 1
            return kept, stats
        except Exception as e:
            return [], {f"error:{type(e).__name__}": 1}


async def main_async(args):
    client = AsyncOpenAI(base_url=os.environ["SYNTH_BASE_URL"], api_key=os.environ.get("SYNTH_API_KEY", "none"))
    model = os.environ["SYNTH_MODEL"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    sem = asyncio.Semaphore(args.concurrency)
    f = open(out / f"sentiment-{args.seed}.jsonl", "a", encoding="utf-8")
    stats, n_texts, n_recs, launched, t0 = {}, 0, 0, 0, time.time()
    by_label = {c: 0 for c in CLASSES}
    pending = set()
    while n_texts < args.n and (pending or launched < args.n):
        while len(pending) < args.concurrency * 2 and launched < args.n:
            pending.add(asyncio.create_task(one_batch(client, model, random.Random(rng.random()), sem)))
            launched += 10
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            kept, st = t.result()
            for k, v in st.items():
                stats[k] = stats.get(k, 0) + v
            for text, label, platform, domain in kept:
                n_texts += 1
                by_label[label] += 1
                for rec in make_records(text, label, platform, domain, rng, f"sent-{args.seed}-{n_texts}"):
                    rec.meta = {"gen_model": model, "gold": label, "platform": platform}
                    f.write(rec.to_json() + "\n")
                    n_recs += 1
            if sum(stats.values()) % 20 == 0:
                print(f"texts={n_texts} records={n_recs} {by_label} {stats} {n_texts / (time.time() - t0) * 3600:.0f} texts/h", flush=True)
    f.close()
    print("done", n_texts, "texts ->", n_recs, "records", by_label, stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth_sentiment")
    ap.add_argument("--n", type=int, default=20000, help="target number of kept texts (records ~1.4x)")
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
