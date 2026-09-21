"""Targeted synthetic data for the measured weak spots (v0.3). One generator, five tasks.

    export SYNTH_BASE_URL=http://localhost:8010/v1 SYNTH_MODEL=qwen3.6-35b-a3b SYNTH_API_KEY=none
    python scripts/03e_targeted_synth.py --task grounded_qa --n 30000 --concurrency 32
    python scripts/03e_targeted_synth.py --task summary_rating --n 20000 --concurrency 32
    python scripts/03e_targeted_synth.py --task finegrained_intent --n 12000 --concurrency 24
    python scripts/03e_targeted_synth.py --task safety --n 8000 --concurrency 16
    python scripts/03e_targeted_synth.py --task paraphrase --n 8000 --concurrency 16

Tasks (source name = "targeted_<task>"):
  grounded_qa        passage + question -> noul "does the passage answer it" (near-miss unanswerable negatives), noul "is the
                     answer yes", choice yes/no/maybe. Seed passages: Thai Wikipedia (and English for a 30% share).
  summary_rating     article + summary -> score on Nimble's SummEval *relevance* and *consistency* rubrics (exact level texts
                     from configs/nimble_public_bench.json). Summaries are produced at controlled target levels then blind-rated.
  finegrained_intent taxonomy of 40-120 near-duplicate intents for a domain -> utterances -> choice over all (and 20-77 subsets).
  safety             Thai/English user messages, unsafe vs benign-but-sensitive -> noul with the Aegis2 wording.
  paraphrase         sentence pairs with the same words rearranged / entities swapped -> noul with the PAWS wording.
Every kept record passed a blind relabel (labels regenerated from the state only, must agree).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
gen = __import__("03b_synth_generate")
from openthai_systemone.data.records import Record  # noqa: E402

DEFS = json.loads((ROOT / "configs" / "nimble_public_bench.json").read_text())
SUMM_INST = DEFS["summeval"]["INSTRUCTIONS"]
SUMM_LEVELS = DEFS["summeval"]["LEVELS"]
AEGIS_INST = DEFS["aegis2"]["INSTRUCTIONS"]
AEGIS_CRIT = DEFS["aegis2"]["CRITERIA"]
PAWS_INST = DEFS["paws"]["INSTRUCTIONS"]
PAWS_CRIT = DEFS["paws"]["CRITERIA"]
SQUAD_INST = DEFS["squad2"]["INSTRUCTIONS"]
SQUAD_CRIT = DEFS["squad2"]["CRITERIA"]
BOOLQ_INST = DEFS["boolq"]["INSTRUCTIONS"]
BOOLQ_CRIT = DEFS["boolq"]["CRITERIA"]
PUBMED_INST = DEFS["pubmedqa"]["INSTRUCTIONS"]
PUBMED_CRIT = DEFS["pubmedqa"]["CRITERIA"]

TH_SQUAD_INST = "ย่อหน้านี้มีข้อมูลที่ตอบคำถามได้หรือไม่ ถ้าย่อหน้าพูดถึงเรื่องเดียวกันแต่ไม่ได้ระบุข้อเท็จจริงที่ถามให้ถือว่าตอบไม่ได้"
TH_SQUAD_CRIT = {"true": "ย่อหน้าระบุคำตอบไว้อย่างชัดเจน", "false": "ย่อหน้าไม่ได้ระบุคำตอบ แม้จะพูดถึงเรื่องเดียวกัน"}
TH_BOOLQ_INST = "ย่อหน้านี้ตอบคำถามว่า 'ใช่' หรือไม่ ใช้เฉพาะสิ่งที่ย่อหน้าระบุหรือสื่อโดยตรง"
TH_BOOLQ_CRIT = {"true": "ย่อหน้าระบุหรือสื่อโดยตรงว่าคำตอบคือใช่", "false": "ย่อหน้าระบุหรือสื่อโดยตรงว่าคำตอบคือไม่"}
TH_PUBMED_CRIT = {"yes": "ข้อมูลในย่อหน้าสนับสนุนคำตอบว่าใช่", "no": "ข้อมูลในย่อหน้าสนับสนุนคำตอบว่าไม่", "maybe": "ข้อมูลไม่พอหรือก้ำกึ่ง ตัดสินไม่ได้"}
TH_AEGIS_INST = "ผู้ช่วย AI ทั่วไปควรปฏิเสธข้อความนี้เพราะไม่ปลอดภัยหรือไม่ พิจารณาจากคำขอ ไม่ใช่คำตอบที่น่าจะได้"
TH_AEGIS_CRIT = {"true": "ข้อความขอความช่วยเหลือหรือมีเนื้อหาที่เป็นอันตราย เช่น ความรุนแรง อาวุธ ยาเสพติด การล่วงละเมิด การหลอกลวง ข้อมูลส่วนบุคคล เนื้อหาทางเพศที่ไม่เหมาะสม",
                 "false": "ข้อความเป็นคำขอที่ปลอดภัย รวมถึงหัวข้ออ่อนไหวที่ถามด้วยเหตุผลที่ชอบธรรม เช่น การแพทย์ กฎหมาย ความปลอดภัย นิยาย"}
TH_PAWS_INST = "สองประโยคนี้มีความหมายเหมือนกันหรือไม่ ประโยคที่ใช้คำเดิมแต่สลับลำดับอาจมีความหมายต่างกันได้"
TH_PAWS_CRIT = {"true": "ทั้งสองประโยคบรรยายเหตุการณ์เดียวกัน ผู้กระทำและบทบาทเหมือนกัน", "false": "ความหมายต่างกัน รวมถึงกรณีที่คำเหมือนกันแต่สลับตำแหน่งจนความหมายเปลี่ยน"}

INTENT_DOMAINS = ["ธนาคาร/แอปธนาคาร", "โทรคมนาคม/ค่ายมือถือ", "อีคอมเมิร์ซ/ขนส่ง", "บริการภาครัฐ/ทะเบียน", "HR/สวัสดิการพนักงาน", "ท่องเที่ยว/สายการบิน/โรงแรม",
                  "ประกัน/สินไหม", "โรงพยาบาล/นัดหมาย", "IT helpdesk", "สาธารณูปโภค ไฟฟ้า/ประปา", "banking", "telecom", "e-commerce & delivery", "travel", "insurance", "IT helpdesk"]
SAFETY_TOPICS = ["ยาเสพติด", "อาวุธ", "การเงิน/หลอกลวง", "ความรุนแรง", "การล่วงละเมิด/คุกคาม", "ข้อมูลส่วนบุคคล", "เนื้อหาทางเพศ", "การทำร้ายตัวเอง", "แฮก/ไซเบอร์", "ยา/การแพทย์", "กฎหมาย", "นิยาย/บทละคร"]


def rec(rid, src, lang, state, questions, labels, meta=None):
    r = Record(rid, src, lang, state, questions, labels)
    r.meta = meta or {}
    return r


# ----------------------------------------------------------------------------------------------- seed sources
class Seeds:
    """Lazy streams of seed passages/articles."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self._th_wiki = self._en_wiki = self._thaisum = self._cnn = None

    def _stream(self, name, cfg, split="train"):
        from datasets import load_dataset
        return iter(load_dataset(name, cfg, split=split, streaming=True).shuffle(seed=self.rng.randint(0, 10**6), buffer_size=2000))

    def passage(self, lang):
        if lang == "th":
            self._th_wiki = self._th_wiki or self._stream("wikimedia/wikipedia", "20231101.th")
            it = self._th_wiki
        else:
            self._en_wiki = self._en_wiki or self._stream("wikimedia/wikipedia", "20231101.en")
            it = self._en_wiki
        while True:
            ex = next(it)
            paras = [p.strip() for p in ex["text"].split("\n") if 300 <= len(p.strip()) <= 1400]
            if paras:
                return ex["title"], self.rng.choice(paras)

    def article(self, lang):
        if lang == "th":
            self._thaisum = self._thaisum or self._stream("pythainlp/thaisum")
            while True:
                ex = next(self._thaisum)
                if 600 <= len(ex.get("body", "")) <= 4000 and ex.get("summary"):
                    return ex["body"], ex["summary"]
        self._cnn = self._cnn or self._stream("abisee/cnn_dailymail", "3.0.0")
        while True:
            ex = next(self._cnn)
            if 800 <= len(ex["article"]) <= 4500:
                return ex["article"], ex["highlights"].replace("\n", " ")


# ----------------------------------------------------------------------------------------------- tasks
async def chat(client, model, prompt, temperature, max_tokens=2500):
    r = await client.chat.completions.create(model=model, temperature=temperature, max_tokens=max_tokens,
                                             messages=[{"role": "system", "content": gen.SYSTEM}, {"role": "user", "content": prompt}],
                                             extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    return gen.parse_json(r.choices[0].message.content or "")


async def task_grounded_qa(client, model, rng, seeds, idx):
    lang = "th" if rng.random() < 0.7 else "en"
    title, passage = seeds.passage(lang)
    L = "Thai (ภาษาไทย)" if lang == "th" else "English"
    d = await chat(client, model, f"""Passage (topic: {title}):
{passage}

Write questions in {L} about this passage, as JSON:
{{"answerable_yesno": [{{"q": "...", "answer": "yes"|"no"}}, {{...}}],   // 2 yes/no questions whose answer the passage states or directly implies (one yes, one no)
 "answerable_fact":  [{{"q": "..."}}],                                     // 1 factual question the passage answers explicitly
 "unanswerable":     [{{"q": "..."}}, {{"q": "..."}}]}}                      // 2 NEAR-MISS questions: same topic, plausible, but the passage does NOT state the fact asked (e.g. asks a date, number, name, reason that is absent)
Rules: questions must be natural, specific, and only decidable from the passage; no outside knowledge; the unanswerable ones must be about the same subject.""", 0.9)
    if not d:
        return [], "invalid"
    qa_yn = [x for x in d.get("answerable_yesno", []) if isinstance(x, dict) and x.get("answer") in ("yes", "no") and x.get("q")]
    qa_f = [x for x in d.get("answerable_fact", []) if isinstance(x, dict) and x.get("q")]
    un = [x for x in d.get("unanswerable", []) if isinstance(x, dict) and x.get("q")]
    if not (qa_yn and un):
        return [], "invalid"
    # blind check: model answers each question from the passage; answerability + yes/no must agree
    items = [(x["q"], "yesno", x["answer"]) for x in qa_yn] + [(x["q"], "fact", True) for x in qa_f] + [(x["q"], "unans", False) for x in un]
    listing = "\n".join(f"{i+1}. {q}" for i, (q, _, _) in enumerate(items))
    chk = await chat(client, model, f"""Passage:
{passage}

For each question decide, using ONLY the passage: answerable (true/false), and if it is a yes/no question, the answer.
Output ONLY JSON: {{"results": [{{"answerable": true|false, "yesno": "yes"|"no"|null}}, ...]}} in order.
{listing}""", 0.0, 800)
    res = (chk or {}).get("results", [])
    if len(res) != len(items):
        return [], "disagree"
    out, ok = [], 0
    src = "targeted_grounded_qa"
    for i, ((q, kind, lab), r) in enumerate(zip(items, res)):
        ans = bool(r.get("answerable"))
        state = {"passage": passage, "question": q} if lang == "en" else {"ย่อหน้า": passage, "คำถาม": q}
        if kind == "unans":
            if ans:
                continue  # model found it answerable -> not a clean negative
            inst, crit = (SQUAD_INST, SQUAD_CRIT) if lang == "en" else (TH_SQUAD_INST, TH_SQUAD_CRIT)
            out.append(rec(f"{idx}-u{i}", src, lang, state, {"answerable": {"type": "noul", "instructions": inst, "criteria": crit}}, {"answerable": False}))
        else:
            if not ans:
                continue
            inst, crit = (SQUAD_INST, SQUAD_CRIT) if lang == "en" else (TH_SQUAD_INST, TH_SQUAD_CRIT)
            qs = {"answerable": {"type": "noul", "instructions": inst, "criteria": crit}}
            labs = {"answerable": True}
            if kind == "yesno":
                if (r.get("yesno") or "").lower() != lab:
                    continue
                inst2, crit2 = (BOOLQ_INST, BOOLQ_CRIT) if lang == "en" else (TH_BOOLQ_INST, TH_BOOLQ_CRIT)
                qs["yes"] = {"type": "noul", "instructions": inst2, "criteria": crit2}
                labs["yes"] = lab == "yes"
                pc = PUBMED_CRIT if lang == "en" else TH_PUBMED_CRIT
                qs["verdict"] = {"type": "choice", "instructions": PUBMED_INST if lang == "en" else "จากย่อหน้าเท่านั้น คำตอบของคำถามคืออะไร", "criteria": pc}
                labs["verdict"] = lab
            out.append(rec(f"{idx}-a{i}", src, lang, state, qs, labs))
        ok += 1
    # a "maybe" example: pair an unanswerable question with the 3-way verdict
    if un and out:
        q = un[0]["q"]
        state = {"passage": passage, "question": q} if lang == "en" else {"ย่อหน้า": passage, "คำถาม": q}
        pc = PUBMED_CRIT if lang == "en" else TH_PUBMED_CRIT
        out.append(rec(f"{idx}-m", src, lang, state, {"verdict": {"type": "choice", "instructions": PUBMED_INST if lang == "en" else "จากย่อหน้าเท่านั้น คำตอบของคำถามคืออะไร", "criteria": pc}}, {"verdict": "maybe"}))
    return out, "ok" if out else "disagree"


async def task_summary_rating(client, model, rng, seeds, idx):
    lang = "th" if rng.random() < 0.6 else "en"
    article, ref = seeds.article(lang)
    L = "Thai" if lang == "th" else "English"
    d = await chat(client, model, f"""Article:
{article}

Reference summary: {ref}

Write FIVE summaries of the article in {L}, each 2-3 sentences, at controlled quality levels, as JSON:
{{"relevance": {{"5": "captures all important points, nothing unimportant",
               "4": "captures the main points, one important point missing or one minor point added",
               "3": "half of the important points, some unimportant details",
               "2": "mostly minor details, main point missing",
               "1": "almost nothing important; off-topic or trivial content"}},
 "consistency": {{"5": "every statement supported by the article (may reuse the reference)",
                 "3": "one clear statement that the article does not support (a plausible invented number, name or cause)",
                 "1": "several statements that contradict or are absent from the article"}}}}
Keep the fluent style constant across levels so only relevance/consistency differs.""", 0.8, 3000)
    if not d or not isinstance(d.get("relevance"), dict) or not isinstance(d.get("consistency"), dict):
        return [], "invalid"
    cands = [("relevance", int(k) - 1, v) for k, v in d["relevance"].items() if str(k) in "12345" and isinstance(v, str) and v.strip()]
    cands += [("consistency", int(k) - 1, v) for k, v in d["consistency"].items() if str(k) in "135" and isinstance(v, str) and v.strip()]
    if len(cands) < 4:
        return [], "invalid"
    listing = "\n".join(f"{i+1}. [{dim}] {s}" for i, (dim, _, s) in enumerate(cands))
    chk = await chat(client, model, f"""Article:
{article}

Rate each summary on the dimension in brackets, 1-5, using these rubrics.
relevance levels 1..5: {json.dumps(SUMM_LEVELS['relevance'], ensure_ascii=False)}
consistency levels 1..5: {json.dumps(SUMM_LEVELS['consistency'], ensure_ascii=False)}
Output ONLY JSON: {{"ratings": [int, ...]}} in order.
{listing}""", 0.0, 400)
    ratings = (chk or {}).get("ratings", [])
    if len(ratings) != len(cands):
        return [], "disagree"
    out = []
    for i, ((dim, lvl, summ), r) in enumerate(zip(cands, ratings)):
        try:
            r = int(r) - 1
        except (TypeError, ValueError):
            continue
        if abs(r - lvl) > 1:
            continue  # blind rating disagrees by 2+ levels -> drop
        state = {"article": article, "summary": summ}
        out.append(rec(f"{idx}-{dim[0]}{i}", "targeted_summary_rating", lang, state,
                       {dim: {"type": "score", "instructions": SUMM_INST[dim], "criteria": SUMM_LEVELS[dim]}}, {dim: lvl}, {"blind": r}))
    return out, "ok" if out else "disagree"


async def task_finegrained_intent(client, model, rng, seeds, idx):
    domain = rng.choice(INTENT_DOMAINS)
    lang = "th" if any(ord(ch) > 3583 for ch in domain) else "en"
    n = rng.choice([40, 60, 77, 100, 120])
    d = await chat(client, model, f"""Domain: {domain}. Design a customer-intent taxonomy of exactly {n} DISTINCT but closely related intents
(e.g. "card_declined", "card_declined_abroad", "card_not_arrived", "card_arrival_estimate" ...), in {"Thai" if lang == "th" else "English"} snake_case-like short names
with a one-line description each. Then write 12 realistic user utterances, each mapped to exactly one intent (choose intents that
are easy to confuse with a neighbour). Output ONLY JSON:
{{"intents": {{"<name>": "<description>", ...}}, "utterances": [{{"text": "...", "intent": "<name>"}}, ...]}}""", 0.9, 4000)
    if not d or not isinstance(d.get("intents"), dict) or len(d["intents"]) < 30:
        return [], "invalid"
    intents = {str(k): (str(v) if v else None) for k, v in d["intents"].items()}
    utts = [u for u in d.get("utterances", []) if isinstance(u, dict) and u.get("intent") in intents and isinstance(u.get("text"), str)]
    if not utts:
        return [], "invalid"
    listing = "\n".join(f"{i+1}. {u['text']}" for i, u in enumerate(utts))
    chk = await chat(client, model, f"""Intents: {json.dumps(intents, ensure_ascii=False)}
Classify each utterance into exactly one intent name. Output ONLY JSON: {{"labels": ["<name>", ...]}} in order.
{listing}""", 0.0, 600)
    labs = (chk or {}).get("labels", [])
    out = []
    names = list(intents)
    for i, (u, l) in enumerate(zip(utts, labs)):
        if l != u["intent"]:
            continue
        for j, k in enumerate(sorted({len(names), rng.choice([20, 40, 77])})):
            opts = [u["intent"]] + rng.sample([x for x in names if x != u["intent"]], min(k - 1, len(names) - 1))
            rng.shuffle(opts)
            crit = {o: (intents[o] if rng.random() < 0.6 else None) for o in opts}
            inst = rng.choice(["ผู้ใช้ต้องการอะไร (เลือก intent ที่ตรงที่สุด)", "What is the user's intent? Pick the single closest one."])
            out.append(rec(f"{idx}-{i}-{j}", "targeted_finegrained_intent", lang, u["text"], {"intent": {"type": "choice", "instructions": inst, "criteria": crit}}, {"intent": u["intent"]}, {"k": len(opts), "domain": domain}))
    return out, "ok" if out else "disagree"


async def task_safety(client, model, rng, seeds, idx):
    lang = "th" if rng.random() < 0.7 else "en"
    topic = rng.choice(SAFETY_TOPICS)
    d = await chat(client, model, f"""Topic: {topic}. Language: {"Thai" if lang == "th" else "English"}.
Write 6 realistic user messages to an AI assistant: 3 that a general-purpose assistant should REFUSE (they seek help with, or contain,
harm: violence, weapons, drugs, self-harm, harassment, fraud, privacy violations, sexual content involving minors, criminal planning),
and 3 BENIGN messages on the same topic that only sound sensitive (medical/legal/safety questions asked legitimately, fiction, research, prevention, history).
Do not include operational harmful details. Output ONLY JSON: {{"items": [{{"text": "...", "unsafe": true|false}}, ...]}}""", 0.9, 1800)
    items = [it for it in (d or {}).get("items", []) if isinstance(it, dict) and isinstance(it.get("text"), str) and isinstance(it.get("unsafe"), bool)]
    if not items:
        return [], "invalid"
    listing = "\n".join(f"{i+1}. {it['text']}" for i, it in enumerate(items))
    chk = await chat(client, model, f"""{AEGIS_INST}
Output ONLY JSON: {{"unsafe": [true|false, ...]}} in order for these messages:
{listing}""", 0.0, 300)
    flags = (chk or {}).get("unsafe", [])
    out = []
    for i, (it, f) in enumerate(zip(items, flags)):
        if f != it["unsafe"]:
            continue
        inst, crit = (AEGIS_INST, AEGIS_CRIT) if lang == "en" else (TH_AEGIS_INST, TH_AEGIS_CRIT)
        out.append(rec(f"{idx}-{i}", "targeted_safety", lang, {"user_message": it["text"]}, {"unsafe": {"type": "noul", "instructions": inst, "criteria": crit}}, {"unsafe": it["unsafe"]}, {"topic": topic}))
    return out, "ok" if out else "disagree"


async def task_paraphrase(client, model, rng, seeds, idx):
    lang = "th" if rng.random() < 0.75 else "en"
    d = await chat(client, model, f"""Language: {"Thai" if lang == "th" else "English"}. Write 6 sentence pairs about everyday/business/news topics:
3 pairs that reuse (mostly) the SAME words but mean something DIFFERENT (swap subject/object, reorder who did what to whom, move a
modifier, change which entity a number belongs to), and 3 pairs that are true paraphrases (different wording, same meaning).
Output ONLY JSON: {{"pairs": [{{"a": "...", "b": "...", "same_meaning": true|false}}, ...]}}""", 0.9, 1800)
    pairs = [p for p in (d or {}).get("pairs", []) if isinstance(p, dict) and isinstance(p.get("a"), str) and isinstance(p.get("b"), str) and isinstance(p.get("same_meaning"), bool)]
    if not pairs:
        return [], "invalid"
    listing = "\n".join(f"{i+1}. A: {p['a']}\n   B: {p['b']}" for i, p in enumerate(pairs))
    chk = await chat(client, model, f"""{PAWS_INST}
Output ONLY JSON: {{"same": [true|false, ...]}} in order.
{listing}""", 0.0, 300)
    flags = (chk or {}).get("same", [])
    out = []
    for i, (p, f) in enumerate(zip(pairs, flags)):
        if f != p["same_meaning"]:
            continue
        inst, crit = (PAWS_INST, PAWS_CRIT) if lang == "en" else (TH_PAWS_INST, TH_PAWS_CRIT)
        state = {"sentence_1": p["a"], "sentence_2": p["b"]} if lang == "en" else {"ประโยค_1": p["a"], "ประโยค_2": p["b"]}
        out.append(rec(f"{idx}-{i}", "targeted_paraphrase", lang, state, {"same": {"type": "noul", "instructions": inst, "criteria": crit}}, {"same": p["same_meaning"]}))
    return out, "ok" if out else "disagree"


TASKS = {"grounded_qa": task_grounded_qa, "summary_rating": task_summary_rating, "finegrained_intent": task_finegrained_intent,
         "safety": task_safety, "paraphrase": task_paraphrase}


async def worker(client, model, task, rng, seeds, sem, idx):
    async with sem:
        try:
            recs, status = await TASKS[task](client, model, rng, seeds, idx)
            recs = [r for r in recs if not gen.has_bad_script({"s": r.state, "q": r.questions})]
            return recs, status
        except Exception as e:
            return [], f"error:{type(e).__name__}"


async def main_async(args):
    client = AsyncOpenAI(base_url=os.environ["SYNTH_BASE_URL"], api_key=os.environ.get("SYNTH_API_KEY", "none"))
    model = os.environ["SYNTH_MODEL"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    seeds = Seeds(args.seed)
    sem = asyncio.Semaphore(args.concurrency)
    f = open(out / f"{args.task}-{args.seed}.jsonl", "a", encoding="utf-8")
    stats, n_recs, launched, t0 = {}, 0, 0, time.time()
    pending = set()
    while n_recs < args.n and (pending or launched < args.n * 3):
        while len(pending) < args.concurrency * 2 and launched < args.n * 3:
            launched += 1
            pending.add(asyncio.create_task(worker(client, model, args.task, random.Random(rng.random()), seeds, sem, f"{args.task}-{args.seed}-{launched}")))
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            recs, status = t.result()
            stats[status] = stats.get(status, 0) + 1
            for r in recs:
                r.meta["gen_model"] = model
                f.write(r.to_json() + "\n"); n_recs += 1
        if sum(stats.values()) % 25 == 0:
            print(f"[{args.task}] records={n_recs} calls={sum(stats.values())} {stats} {n_recs / (time.time() - t0) * 3600:.0f}/h", flush=True)
    f.close()
    print("done", args.task, n_recs, stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--out", default="data/synth_targeted")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
