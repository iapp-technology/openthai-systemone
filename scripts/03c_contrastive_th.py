"""Thai contrastive holdout: pairs of states that differ in ONE relevant fact so the answer flips (Nimble-style).

    export SYNTH_BASE_URL=http://localhost:8000/v1 SYNTH_MODEL=<any-openai-compatible-chat-model> SYNTH_API_KEY=none
    python scripts/03c_contrastive_th.py --out data/decision/contrastive_th.eval.jsonl --pairs 300

Each pair shares the question (choice/score/noul) and differs only in the one fact; both sides are blind-relabelled
by the generator model and kept only when the relabel agrees on BOTH sides and the two labels differ.
Output is eval-only (split="eval") and is never used for training.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
gen = __import__("03b_synth_generate")
from openthai_systemone.data.records import Record, write_jsonl  # noqa: E402

PAIR_TEMPLATE = """Domain: {domain}
Language: Thai (ภาษาไทย) for everything. State style: {style}.
Question type: {qtype}. {qhint}

Produce ONE question and TWO states (state_a, state_b) that are identical except for ONE relevant fact, so that the
correct answer differs between the two states. Everything else (names, numbers, distractors, wording) must be the same.
Output ONLY JSON:
{{
  "question": {{"type": "{qtype}", "instructions": "...", "criteria": ...}},
  "state_a": <string or JSON object>, "label_a": <answer for state_a>,
  "state_b": <string or JSON object>, "label_b": <answer for state_b>,
  "changed_fact": "<one sentence describing the difference>"
}}
STRICT RULES: answers must be derivable from the state alone; option descriptions describe meaning only and must not
hint at which is correct; use only Thai, English, digits and punctuation; label_a != label_b."""

QHINTS = {
    "choice": "Use between 3 and {n} options as a criteria object (option -> description or null).",
    "score": "Use {n} ordered levels (array of descriptions, low -> high); labels are level indexes (int).",
    "noul": "criteria = {{\"true\": \"...\", \"false\": \"...\"}}; labels are true/false.",
}


async def one_pair(client, model, rng, sem):
    async with sem:
        domain = rng.choice(gen.DOMAINS)
        style = rng.choice(gen.STATE_STYLES)
        qtype = rng.choices(["choice", "score", "noul"], weights=[0.5, 0.2, 0.3])[0]
        n = rng.randint(3, 12) if qtype == "choice" else rng.randint(3, 7)
        prompt = PAIR_TEMPLATE.format(domain=domain, style=style, qtype=qtype, qhint=QHINTS[qtype].format(n=n))
        try:
            r = await client.chat.completions.create(model=model, temperature=0.9, max_tokens=3000,
                                                     messages=[{"role": "system", "content": gen.SYSTEM}, {"role": "user", "content": prompt}],
                                                     extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            d = gen.parse_json(r.choices[0].message.content or "")
            if not d or "question" not in d:
                return None, "invalid"
            q = d["question"]
            recs = []
            for side in ("a", "b"):
                rec = {"state": d.get(f"state_{side}"), "questions": {"decision": q}, "labels": {"decision": d.get(f"label_{side}")}}
                if not gen.validate(rec) or gen.has_bad_script(rec):
                    return None, "invalid"
                recs.append(rec)
            if recs[0]["labels"]["decision"] == recs[1]["labels"]["decision"]:
                return None, "same_label"
            for rec in recs:  # blind relabel must agree
                r2 = await client.chat.completions.create(model=model, temperature=0.0, max_tokens=256,
                                                          messages=[{"role": "system", "content": gen.SYSTEM}, {"role": "user", "content": gen.LABEL_TEMPLATE.format(
                                                              state=rec["state"] if isinstance(rec["state"], str) else json.dumps(rec["state"], ensure_ascii=False),
                                                              questions=json.dumps(rec["questions"], ensure_ascii=False))}],
                                                          extra_body={"chat_template_kwargs": {"enable_thinking": False}})
                lab2 = (gen.parse_json(r2.choices[0].message.content or "") or {}).get("labels", {})
                if gen.normalise_label(q, lab2.get("decision")) != rec["labels"]["decision"]:
                    return None, "disagree"
            return {"domain": domain, "style": style, "qtype": qtype, "changed_fact": d.get("changed_fact", ""), "sides": recs}, "ok"
        except Exception as e:
            return None, f"error:{type(e).__name__}"


async def main_async(args):
    client = AsyncOpenAI(base_url=os.environ["SYNTH_BASE_URL"], api_key=os.environ.get("SYNTH_API_KEY", "none"))
    model = os.environ["SYNTH_MODEL"]
    rng = random.Random(args.seed)
    sem = asyncio.Semaphore(args.concurrency)
    stats, pairs = {}, []
    launched, target_launch = 0, int(args.pairs / 0.25)
    pending = set()
    while len(pairs) < args.pairs and (pending or launched < target_launch):
        while len(pending) < args.concurrency * 2 and launched < target_launch:
            pending.add(asyncio.create_task(one_pair(client, model, random.Random(rng.random()), sem)))
            launched += 1
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for t in done:
            p, status = t.result()
            stats[status] = stats.get(status, 0) + 1
            if p:
                pairs.append(p)
        if sum(stats.values()) % 50 == 0:
            print(f"pairs={len(pairs)} {stats}", flush=True)
    recs = []
    for i, p in enumerate(pairs[: args.pairs]):
        for side, rec in zip("ab", p["sides"]):
            recs.append(Record(f"contrastive-th-{args.seed}-{i}{side}", "contrastive_th", "th", rec["state"], rec["questions"], rec["labels"],
                               split="eval", meta={"pair": i, "domain": p["domain"], "qtype": p["qtype"], "changed_fact": p["changed_fact"], "gen_model": model}))
    n = write_jsonl(args.out, recs)
    print("wrote", n, "records", stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/decision/contrastive_th.eval.jsonl")
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
