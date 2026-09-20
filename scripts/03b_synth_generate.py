"""Synthetic decision-data generation through an OpenAI-compatible endpoint (the H100 server vLLM gateway).

    export SYNTH_BASE_URL=http://localhost:8000/v1  SYNTH_MODEL=<any-openai-compatible-chat-model>  SYNTH_API_KEY=<key-or-none>
    python scripts/03b_synth_generate.py --out data/synth --n 600000 --concurrency 32

Each generated record (jsonl):
    {"id", "lang", "domain", "state": str|dict, "questions": {qid: Choice|Score|Noul}, "labels": {qid: label}, "source": "synth"}

Quality gate (self-consistency): a second, independent call labels the generated (state, questions) blind; the
record is kept only if the two label sets agree on every question.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --------------------------------------------------------------------------- taxonomy
DOMAINS = [
    "customer support ticket routing", "content moderation of social posts", "spam / phishing email detection",
    "e-commerce product categorisation", "e-commerce order status & returns", "banking transaction categorisation",
    "KYC / form field validation", "loan application triage", "insurance claim triage", "healthcare symptom triage",
    "appointment scheduling intents", "HR resume screening", "employee leave / expense requests", "IT helpdesk tickets",
    "cybersecurity alert triage", "legal document type classification", "government e-service requests (Thai)",
    "tax / invoice document fields", "logistics & delivery exceptions", "restaurant & food delivery reviews",
    "hotel & travel booking intents", "real-estate listing attributes", "job posting attributes", "news topic classification",
    "news stance & sentiment", "product review sentiment & aspects", "chat intent detection (LINE / messenger)",
    "call-centre transcript disposition", "sales lead qualification", "CRM next-best-action",
    "web UI action selection (accessibility tree)", "mobile app UI action selection", "desktop app UI action selection",
    "browser agent: which element to click / type / select", "tool / function selection for an agent",
    "agent step verification (did the action succeed?)", "prompt-injection / unsafe instruction detection",
    "RAG passage relevance judging", "answer correctness / hallucination judging", "code review severity",
    "CI failure categorisation", "log line anomaly classification", "database record deduplication (same entity?)",
    "address & name normalisation (Thai)", "education: exam answer grading rubric", "education: question difficulty",
    "language identification & code-switching", "translation quality rating", "summary quality rating",
    "toxicity & harassment levels", "age-appropriateness rating", "medical report ICD-style coding",
    "supply-chain PO matching", "meeting notes: action item extraction (is this an action item?)",
    "smart-home / IoT command routing", "vehicle telematics event classification", "drone / robot manoeuvre selection",
    "game state action selection", "sports commentary event tagging", "music / media genre tagging",
]
LANGS = [("th", 0.55), ("en", 0.30), ("mixed", 0.15)]
STATE_STYLES = ["prose", "json", "key-value lines", "chat transcript", "table-like text", "accessibility tree / DOM dump"]


def sample_option_count(rng: random.Random) -> int:
    """log-uniform on 2..255, with a bump at small counts (most real tasks have < 20 options)."""
    if rng.random() < 0.6:
        return rng.randint(2, 12)
    return int(math.exp(rng.uniform(math.log(2), math.log(255))))


SYSTEM = (
    "You generate training data for a decision model. Output ONLY a JSON object, no prose, no markdown fences. "
    "All text must be natural and realistic; Thai text must be fluent native Thai."
)

GEN_TEMPLATE = """Domain: {domain}
Language of the state and questions: {lang_desc}
State style: {style}
Number of questions: {n_q} (types: {types})
For each "choice" question use exactly {n_opts} options; option names are short strings; descriptions may be null for some options.
For each "score" question use {n_levels} ordered levels (low -> high) as an array of descriptions.
For "noul" questions optionally include criteria {{"true": "...", "false": "..."}}.

Produce:
{{
  "state": <string, or a JSON object/array if the style is json>,
  "questions": {{
     "<qid>": {{"type": "choice", "instructions": "...", "criteria": {{"<option>": "<description or null>", ...}}}},
     "<qid>": {{"type": "score",  "instructions": "...", "criteria": ["<level 0>", "<level 1>", ...]}},
     "<qid>": {{"type": "noul",   "instructions": "...", "criteria": {{"true": "...", "false": "..."}} }}
  }},
  "labels": {{ "<qid>": <option name | level index (int) | true/false> }}
}}
The state must contain enough evidence to decide every question unambiguously, but should not be trivial:
include realistic noise, distractors and irrelevant details. Make the correct option NOT the first one listed
in at least half of the cases.
STRICT RULES:
- Every answer must be derivable from the state alone. No outside/world knowledge questions (no postcodes, dates, trivia).
- Option descriptions must describe what the option MEANS, never whether it is correct. Never write things like
  "no supporting data", "not mentioned", "the correct one", "as stated in the state" in any description. All options
  must get the same style and length of description (or all null).
- Use only Thai, English, digits and punctuation. No characters from other scripts. Spell Thai words correctly.
{extra}"""

LABEL_TEMPLATE = """Given the state and the questions, answer every question. Output ONLY JSON:
{{"labels": {{"<qid>": <option name | level index (int) | true/false>}}}}

STATE:
{state}

QUESTIONS:
{questions}"""

BLIND_TEMPLATE = """Answer every question using ONLY the question text and option descriptions (there is no state).
Guess if needed. Output ONLY JSON: {{"labels": {{"<qid>": <option name | level index (int) | true/false>}}}}

QUESTIONS:
{questions}"""

GEN_VERSION = 2

# allowed: Thai, basic Latin + Latin-1 supplement + Latin extended, general punctuation, currency, arrows, box drawing,
# math symbols, emoji planes. Anything else (CJK, Bengali, Devanagari, Cyrillic, Arabic, Hangul...) rejects the record.
_bad_script_re = re.compile(
    r"[\u0400-\u052F\u0590-\u08FF\u0900-\u0DFF\u1100-\u11FF\u3040-\u30FF\u3130-\u318F\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF]"
)


def has_bad_script(obj) -> bool:
    return bool(_bad_script_re.search(json.dumps(obj, ensure_ascii=False)))


LANG_DESC = {
    "th": "Thai (ภาษาไทย) for everything, including option names and descriptions",
    "en": "English",
    "mixed": "Thai state with English option names / instructions (or vice versa), as commonly seen in Thai software",
}

EXTRA_HINTS = [
    "",
    "Make the state long (300-600 words).",
    "Use realistic Thai names, addresses, phone formats and Buddhist-era dates where appropriate.",
    "Include an option that is a near-miss distractor of the correct one.",
    "Some option descriptions should be null.",
    "The state is a UI accessibility tree: lines like `[12] button \"ชำระเงิน\"`, `[13] textbox \"อีเมล\" value=\"\"`; the choice options are element ids or actions.",
    "The state is a tool-calling agent scratchpad with the user request and available tools; the choice is which tool to call next.",
]


def build_gen_prompt(rng: random.Random, langs=None):
    domain = rng.choice(DOMAINS)
    pool = [(l, w) for l, w in LANGS if not langs or l in langs]
    lang = rng.choices([l for l, _ in pool], weights=[w for _, w in pool])[0]
    style = rng.choice(STATE_STYLES)
    n_q = rng.choice([1, 1, 2, 2, 3, 4])
    types = rng.choices(["choice", "score", "noul"], weights=[0.55, 0.2, 0.25], k=n_q)
    n_opts = sample_option_count(rng)
    n_levels = rng.randint(2, 10)
    extra = rng.choice(EXTRA_HINTS)
    if "UI" in domain or "browser" in domain or "element" in domain:
        extra = EXTRA_HINTS[5]
    if "tool" in domain:
        extra = EXTRA_HINTS[6]
    prompt = GEN_TEMPLATE.format(
        domain=domain, lang_desc=LANG_DESC[lang], style=style, n_q=n_q, types=", ".join(types),
        n_opts=n_opts, n_levels=n_levels, extra=extra,
    )
    return prompt, {"domain": domain, "lang": lang, "style": style, "n_opts": n_opts}


_json_re = re.compile(r"\{.*\}", re.S)


def parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    m = _json_re.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def normalise_label(q, lab):
    t = q.get("type")
    if t == "choice":
        return str(lab) if lab is not None else None
    if t == "score":
        try:
            return int(lab)
        except (TypeError, ValueError):
            return None
    if t == "noul":
        if isinstance(lab, bool):
            return lab
        if isinstance(lab, str):
            return lab.strip().lower() in {"true", "yes", "ใช่", "1"}
        return bool(lab) if lab is not None else None
    return None


def validate(rec) -> bool:
    from openthai_systemone.types import parse_question

    if not isinstance(rec, dict) or "state" not in rec or not isinstance(rec.get("questions"), dict):
        return False
    labels = rec.get("labels") or {}
    if set(labels) != set(rec["questions"]) or not labels:
        return False
    for qid, q in rec["questions"].items():
        try:
            pq = parse_question(q)
        except Exception:
            return False
        lab = normalise_label(q, labels[qid])
        if lab is None:
            return False
        if pq.type == "choice" and lab not in pq.criteria:
            return False
        if pq.type == "score" and not (0 <= lab < len(pq.criteria)):
            return False
        labels[qid] = lab
    rec["labels"] = labels
    return True


async def one(client, model, rng, sem, temperature, langs=None):
    async with sem:
        prompt, meta = build_gen_prompt(rng, langs)
        try:
            r = await client.chat.completions.create(
                model=model, temperature=temperature, max_tokens=4096,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            rec = parse_json(r.choices[0].message.content or "")
            if not rec or not validate(rec):
                return None, "invalid"
            # self-consistency relabel
            r2 = await client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=512,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": LABEL_TEMPLATE.format(
                    state=json.dumps(rec["state"], ensure_ascii=False) if not isinstance(rec["state"], str) else rec["state"],
                    questions=json.dumps(rec["questions"], ensure_ascii=False))}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            lab2 = parse_json(r2.choices[0].message.content or "") or {}
            lab2 = lab2.get("labels", {})
            for qid, q in rec["questions"].items():
                if normalise_label(q, lab2.get(qid)) != rec["labels"][qid]:
                    return None, "disagree"
            if has_bad_script(rec):
                return None, "bad_script"
            # state-blind leakage check: if the labeler gets everything right without the state, the questions
            # are answerable from descriptions / world knowledge alone -> not a System-One example
            chance = 1.0
            for q in rec["questions"].values():
                chance *= 1.0 / max(2, len(q["criteria"]) if isinstance(q.get("criteria"), (dict, list)) else 2)
            if chance < 0.34:
                r3 = await client.chat.completions.create(
                    model=model, temperature=0.0, max_tokens=512,
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": BLIND_TEMPLATE.format(
                        questions=json.dumps(rec["questions"], ensure_ascii=False))}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                lab3 = (parse_json(r3.choices[0].message.content or "") or {}).get("labels", {})
                if all(normalise_label(q, lab3.get(qid)) == rec["labels"][qid] for qid, q in rec["questions"].items()):
                    return None, "leak"
            rec.update(meta)
            rec["source"] = "synth"
            rec["gen_version"] = GEN_VERSION
            rec["gen_model"] = model
            return rec, "ok"
        except Exception as e:  # network / server errors
            return None, f"error:{type(e).__name__}"


async def main_async(args):
    client = AsyncOpenAI(base_url=os.environ["SYNTH_BASE_URL"], api_key=os.environ.get("SYNTH_API_KEY", "none"))
    model = os.environ["SYNTH_MODEL"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    sem = asyncio.Semaphore(args.concurrency)
    langs = [l for l in args.lang.split(",") if l] or None
    stats = {}
    kept = 0
    shard = 0
    t0 = time.time()
    f = open(out / f"synth-{args.seed}-{shard:04d}.jsonl", "a", encoding="utf-8")
    pending = set()
    launched = 0
    target_launch = int(args.n / max(args.expected_yield, 0.05))
    while kept < args.n and (pending or launched < target_launch):
        while len(pending) < args.concurrency * 2 and launched < target_launch:
            pending.add(asyncio.create_task(one(client, model, random.Random(rng.random()), sem, args.temperature, langs)))
            launched += 1
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            rec, status = t.result()
            stats[status] = stats.get(status, 0) + 1
            if rec:
                rec["id"] = f"synth-{args.seed}-{kept}"
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                if kept % args.shard_size == 0:
                    f.close()
                    shard += 1
                    f = open(out / f"synth-{args.seed}-{shard:04d}.jsonl", "a", encoding="utf-8")
            if sum(stats.values()) % 100 == 0:
                rate = kept / (time.time() - t0)
                print(f"kept={kept} launched={launched} {stats} {rate * 3600:.0f}/h", flush=True)
    f.close()
    print("done", kept, stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--n", type=int, default=600_000)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=10_000)
    ap.add_argument("--expected-yield", type=float, default=0.6)
    ap.add_argument("--lang", default="", help="comma list of th,en,mixed to restrict languages (default: all, weighted)")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
