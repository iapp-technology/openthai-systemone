"""Stage 2 data: convert public datasets into unified decision records (jsonl).

    python scripts/03_decision_data.py --out data/decision --limit 2000        # smoke test
    python scripts/03_decision_data.py --out data/decision                     # full (run on the H100 server)
    python scripts/03_decision_data.py --only wisesight,massive_th --out data/decision

Held-out-for-eval datasets (never in train): wisesight, sib200_th, banking77, mind2web(test_*), xlam(eval slice).
Each converter yields Record objects; the driver writes <out>/<source>.<split>.jsonl and a manifest.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openthai_systemone.data.records import Record, choice_q, noul_q, pick_template, score_q, write_jsonl  # noqa: E402

REGISTRY: Dict[str, Callable[..., Iterator[Record]]] = {}
EVAL_ONLY = {"wisesight", "sib200_th", "banking77"}


def converter(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def load(name, config=None, split="train", **kw):
    from datasets import load_dataset
    return load_dataset(name, config, split=split, streaming=True, **kw)


def label_name(ds, ex, col="label", text_col="label_text"):
    """Resolve a class label to its string name across parquet/ClassLabel variants."""
    if text_col in ex and ex[text_col] is not None:
        return str(ex[text_col])
    v = ex[col]
    if isinstance(v, str):
        return v
    feat = getattr(ds, "features", None) or {}
    f = feat.get(col)
    if f is not None and hasattr(f, "int2str"):
        return f.int2str(int(v))
    return str(v)


def take(it, limit):
    for i, x in enumerate(it):
        if limit and i >= limit:
            break
        yield x


# ----------------------------------------------------------------------------- Thai
@converter("wisesight")
def wisesight(limit, rng):
    names = {"pos": "เชิงบวก", "neu": "เป็นกลาง", "neg": "เชิงลบ", "q": "คำถาม"}
    ds = load("pythainlp/wisesight_sentiment", split="test")
    for i, ex in enumerate(take(ds, limit)):
        lab = label_name(ds, ex, "category")
        yield Record(f"wisesight-{i}", "wisesight", "th", ex["texts"],
                     {"sentiment": choice_q("ข้อความนี้แสดงความรู้สึกแบบใด", list(names), {k: v for k, v in names.items()})},
                     {"sentiment": lab}, split="eval")


@converter("wongnai")
def wongnai(limit, rng):
    for split in ("train", "test"):
        ds = load("Wongnai/wongnai_reviews", split=split)
        for i, ex in enumerate(take(ds, limit)):
            stars = int(ex["star_rating"])  # 0..4
            levels = ["แย่มาก (1 ดาว)", "แย่ (2 ดาว)", "ปานกลาง (3 ดาว)", "ดี (4 ดาว)", "ดีมาก (5 ดาว)"]
            yield Record(f"wongnai-{split}-{i}", "wongnai", "th", ex["review_body"],
                         {"rating": score_q(rng.choice(["รีวิวนี้ให้คะแนนร้านกี่ดาว", "ประเมินความพึงพอใจของผู้รีวิว"]), levels)},
                         {"rating": stars}, split="train" if split == "train" else "eval")


@converter("prachathai")
def prachathai(limit, rng):
    topics = {"politics": "การเมือง", "human_rights": "สิทธิมนุษยชน", "quality_of_life": "คุณภาพชีวิต", "international": "ต่างประเทศ",
              "social": "สังคม", "environment": "สิ่งแวดล้อม", "economics": "เศรษฐกิจ", "culture": "วัฒนธรรม", "labor": "แรงงาน",
              "national_security": "ความมั่นคง", "ict": "ไอซีที", "education": "การศึกษา"}
    for split in ("train", "validation"):
        ds = load("PyThaiNLP/prachathai67k", split=split, revision="refs/convert/parquet")
        for i, ex in enumerate(take(ds, limit)):
            state = {"title": ex["title"], "body": ex["body_text"][:3000]}
            pos = [t for t in topics if ex.get(t)]
            qs, labs = {}, {}
            # per-topic noul for 2 random topics (one positive if any)
            cand = (rng.sample(pos, 1) if pos else []) + rng.sample([t for t in topics if t not in pos], 1)
            for t in cand:
                qs[f"is_{t}"] = noul_q(f"ข่าวนี้เกี่ยวข้องกับหัวข้อ '{topics[t]}' หรือไม่")
                labs[f"is_{t}"] = t in pos
            if len(pos) == 1:
                qs["topic"] = choice_q("ข่าวนี้อยู่ในหมวดหลักใด", list(topics.values()))
                labs["topic"] = topics[pos[0]]
            yield Record(f"prachathai-{split}-{i}", "prachathai", "th", state, qs, labs,
                         split="train" if split == "train" else "eval")


@converter("xnli_th")
def xnli_th(limit, rng):
    for split in ("train", "validation"):
        ds = load("facebook/xnli", "th", split=split)
        for i, ex in enumerate(take(ds, limit)):
            lab = ["entailment", "neutral", "contradiction"][int(ex["label"])]
            opts = {"entailment": "ประโยคที่สองสรุปได้จากประโยคแรก", "neutral": "ไม่สามารถสรุปได้", "contradiction": "ประโยคที่สองขัดแย้งกับประโยคแรก"}
            yield Record(f"xnli-th-{split}-{i}", "xnli_th", "th", {"premise": ex["premise"], "hypothesis": ex["hypothesis"]},
                         {"nli": choice_q("ความสัมพันธ์ระหว่าง premise และ hypothesis คืออะไร", list(opts), opts),
                          "entails": noul_q("hypothesis สรุปได้จาก premise หรือไม่")},
                         {"nli": lab, "entails": lab == "entailment"}, split="train" if split == "train" else "eval")


@converter("massive_th")
def massive_th(limit, rng):
    for split in ("train", "validation", "test"):
        ds = load("mteb/amazon_massive_intent", "th", split=split)
        rows = list(take(ds, limit))
        intents = sorted({label_name(ds, ex) for ex in rows})
        if len(intents) < 2:
            continue
        for i, ex in enumerate(rows):
            correct = label_name(ds, ex)
            k = min(len(intents), rng.choice([len(intents), 20, 10, 5]))
            opts = [correct] + rng.sample([x for x in intents if x != correct], max(0, min(k - 1, len(intents) - 1)))
            rng.shuffle(opts)
            yield Record(f"massive-th-{split}-{i}", "massive_th", "mixed", ex["text"],
                         {"intent": choice_q(rng.choice(["ผู้ใช้ต้องการทำอะไร (intent)", "What does the user want to do?"]),
                                             [o.replace("_", " ") for o in opts])},
                         {"intent": correct.replace("_", " ")}, split="train" if split == "train" else "eval")


@converter("massive_en")
def massive_en(limit, rng):
    ds = load("mteb/amazon_massive_intent", "en", split="train")
    rows = list(take(ds, limit))
    intents = sorted({label_name(ds, ex) for ex in rows})
    for i, ex in enumerate(rows):
        correct = label_name(ds, ex)
        k = min(len(intents), rng.choice([len(intents), 30, 10]))
        opts = [correct] + rng.sample([x for x in intents if x != correct], max(0, min(k - 1, len(intents) - 1)))
        rng.shuffle(opts)
        yield Record(f"massive-en-{i}", "massive_en", "en", ex["text"],
                     {"intent": choice_q(pick_template(rng, "en", "choice", "intent"), [o.replace("_", " ") for o in opts])},
                     {"intent": correct.replace("_", " ")})


@converter("sib200_th")
def sib200_th(limit, rng):
    ds = load("Davlan/sib200", "tha_Thai", split="test")
    cats = ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]
    for i, ex in enumerate(take(ds, limit)):
        yield Record(f"sib200-th-{i}", "sib200_th", "mixed", ex["text"],
                     {"topic": choice_q("ข้อความนี้เกี่ยวกับหัวข้อใด", cats)}, {"topic": ex["category"]}, split="eval")


@converter("thai_toxicity")
def thai_toxicity(limit, rng):
    ds = load("tmu-nlp/thai_toxicity_tweet", split="train", revision="refs/convert/parquet")
    for i, ex in enumerate(take(ds, limit)):
        if not ex.get("tweet_text") or ex["tweet_text"] == "TWEET_NOT_FOUND":
            continue
        yield Record(f"thaitox-{i}", "thai_toxicity", "th", ex["tweet_text"],
                     {"toxic": noul_q("ข้อความนี้เป็นข้อความที่เป็นพิษ (toxic) หรือไม่", "มีการด่าทอ เหยียดหยาม หรือคุกคาม", "สุภาพหรือเป็นกลาง")},
                     {"toxic": bool(int(ex["is_toxic"]))})


@converter("thai_exam")
def thai_exam(limit, rng):
    for cfg in ("onet", "ic", "tgat", "tpat1", "a_level"):
        try:
            ds = load("scb10x/thai_exam", cfg, split="test")
        except Exception:
            continue
        for i, ex in enumerate(take(ds, limit)):
            opts = {k: ex[k] for k in "abcde" if ex.get(k)}
            if len(opts) < 2 or ex.get("answer") not in opts:
                continue
            yield Record(f"thaiexam-{cfg}-{i}", "thai_exam", "th", ex["question"],
                         {"answer": choice_q("ข้อใดคือคำตอบที่ถูกต้อง", list(opts), opts)}, {"answer": ex["answer"]},
                         split="train")


@converter("iapp_wiki_qa")
def iapp_wiki_qa(limit, rng):
    ds = load("iapp/iapp_wiki_qa_squad", split="train")
    rows = list(take(ds, limit or 20000))
    for i, ex in enumerate(rows):
        neg = rows[rng.randrange(len(rows))]
        for j, (ctx, lab) in enumerate([(ex["context"], True), (neg["context"], neg["context"] == ex["context"])]):
            yield Record(f"iappqa-{i}-{j}", "iapp_wiki_qa", "th", {"question": ex["question"], "passage": ctx[:2000]},
                         {"relevant": noul_q("passage นี้มีคำตอบของคำถามหรือไม่")}, {"relevant": lab})


# ----------------------------------------------------------------------------- English / multilingual
@converter("banking77")
def banking77(limit, rng):
    ds = load("mteb/banking77", split="test")
    rows = list(take(ds, limit))
    names = sorted({label_name(ds, ex) for ex in rows})
    for i, ex in enumerate(rows):
        yield Record(f"banking77-{i}", "banking77", "en", ex["text"],
                     {"intent": choice_q("What is the customer's intent?", [n.replace("_", " ") for n in names])},
                     {"intent": label_name(ds, ex).replace("_", " ")}, split="eval")


@converter("clinc")
def clinc(limit, rng):
    ds = load("clinc/clinc_oos", "plus", split="train")
    rows = list(take(ds, limit))
    names = sorted({label_name(ds, ex, "intent") for ex in rows})
    for i, ex in enumerate(rows):
        correct = label_name(ds, ex, "intent")
        k = min(len(names), rng.choice([len(names), 50, 15]))
        opts = [correct] + rng.sample([x for x in names if x != correct], max(0, min(k - 1, len(names) - 1)))
        rng.shuffle(opts)
        yield Record(f"clinc-{i}", "clinc", "en", ex["text"],
                     {"intent": choice_q(pick_template(rng, "en", "choice", "intent"), opts)}, {"intent": correct})


@converter("ag_news")
def ag_news(limit, rng):
    ds = load("fancyzhx/ag_news", split="train")
    names = ["World", "Sports", "Business", "Sci/Tech"]
    for i, ex in enumerate(take(ds, limit)):
        yield Record(f"agnews-{i}", "ag_news", "en", ex["text"],
                     {"topic": choice_q(pick_template(rng, "en", "choice", "news topic"), names)}, {"topic": names[ex["label"]]})


@converter("dbpedia")
def dbpedia(limit, rng):
    ds = load("fancyzhx/dbpedia_14", split="train")
    names = ds.features["label"].names
    for i, ex in enumerate(take(ds, limit)):
        yield Record(f"dbpedia-{i}", "dbpedia", "en", {"title": ex["title"], "content": ex["content"]},
                     {"type": choice_q("What kind of entity is this article about?", names)}, {"type": names[ex["label"]]})


@converter("mnli")
def mnli(limit, rng):
    ds = load("nyu-mll/glue", "mnli", split="train")
    for i, ex in enumerate(take(ds, limit)):
        lab = ["entailment", "neutral", "contradiction"][int(ex["label"])]
        yield Record(f"mnli-{i}", "mnli", "en", {"premise": ex["premise"], "hypothesis": ex["hypothesis"]},
                     {"nli": choice_q("What is the relation between premise and hypothesis?", ["entailment", "neutral", "contradiction"]),
                      "entails": noul_q("Does the premise entail the hypothesis?")},
                     {"nli": lab, "entails": lab == "entailment"})


@converter("mmlu")
def mmlu(limit, rng):
    ds = load("cais/mmlu", "all", split="auxiliary_train")
    for i, ex in enumerate(take(ds, limit)):
        opts = {f"{c}": None for c in ex["choices"]}
        if len(opts) != len(ex["choices"]):
            continue
        yield Record(f"mmlu-{i}", "mmlu", "en", ex["question"],
                     {"answer": choice_q("Which option answers the question?", list(opts))},
                     {"answer": ex["choices"][int(ex["answer"])]})


@converter("arc")
def arc(limit, rng):
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        ds = load("allenai/ai2_arc", cfg, split="train")
        for i, ex in enumerate(take(ds, limit)):
            texts = ex["choices"]["text"]
            labels = ex["choices"]["label"]
            if ex["answerKey"] not in labels or len(set(texts)) != len(texts):
                continue
            yield Record(f"arc-{cfg}-{i}", "arc", "en", ex["question"],
                         {"answer": choice_q("Which option is correct?", texts)}, {"answer": texts[labels.index(ex["answerKey"])]})


@converter("helpsteer2")
def helpsteer2(limit, rng):
    ds = load("nvidia/HelpSteer2", split="train")
    dims = {"helpfulness": "How helpful is the response?", "correctness": "How correct is the response?",
            "coherence": "How coherent is the response?", "verbosity": "How verbose is the response?"}
    for i, ex in enumerate(take(ds, limit)):
        qs, labs = {}, {}
        for d, inst in dims.items():
            if ex.get(d) is None:
                continue
            qs[d] = score_q(inst, ["0 (worst)", "1", "2", "3", "4 (best)"])
            labs[d] = int(ex[d])
        yield Record(f"helpsteer2-{i}", "helpsteer2", "en", {"prompt": ex["prompt"][:3000], "response": ex["response"][:4000]}, qs, labs)


# ----------------------------------------------------------------------------- agents / computer use
def _cand_text(c) -> str:
    try:
        attrs = json.loads(c.get("attributes") or "{}")
    except Exception:
        attrs = {}
    keep = {k: v for k, v in attrs.items() if k in ("id", "class", "name", "type", "role", "aria_label", "aria-label", "title", "placeholder", "value", "alt", "href")}
    label = keep.pop("aria_label", None) or keep.pop("aria-label", None) or keep.pop("title", None) or ""
    parts = [c.get("tag", "")] + ([label] if label else []) + [f'{k}="{str(v)[:40]}"' for k, v in list(keep.items())[:4]]
    return " ".join(p for p in parts if p)[:160]


@converter("mind2web")
def mind2web(limit, rng):
    ds = load("osunlp/Mind2Web", split="train")
    for i, ex in enumerate(take(ds, limit)):
        history = []
        for j, act in enumerate(ex["actions"]):
            pos = act.get("pos_candidates") or []
            neg = act.get("neg_candidates") or []
            if not pos:
                continue
            target = pos[0]
            k = rng.choice([8, 16, 32, 64])
            cands = [target] + rng.sample(neg, min(len(neg), k - 1))
            rng.shuffle(cands)
            opts = {}
            for c in cands:
                key = f"[{c.get('backend_node_id')}] {_cand_text(c)}"
                opts[key] = None
            tkey = f"[{target.get('backend_node_id')}] {_cand_text(target)}"
            if tkey not in opts:
                continue
            op = (act.get("operation") or {}).get("op", "CLICK")
            state = {"task": ex["confirmed_task"], "website": ex["website"], "previous_actions": history[-5:],
                     "candidate_elements": list(opts)}
            yield Record(f"mind2web-{i}-{j}", "mind2web", "en", state,
                         {"target": choice_q("Which element should the agent act on next?", list(opts)),
                          "operation": choice_q("Which operation should be performed on it?", ["CLICK", "TYPE", "SELECT"])},
                         {"target": tkey, "operation": op if op in ("CLICK", "TYPE", "SELECT") else "CLICK"})
            history.append(f"{op} {_cand_text(target)}")


@converter("xlam_tools")
def xlam_tools(limit, rng):
    ds = load("lockon/xlam-function-calling-60k", split="train")
    for i, ex in enumerate(take(ds, limit)):
        try:
            tools = json.loads(ex["tools"]) if isinstance(ex["tools"], str) else ex["tools"]
            answers = json.loads(ex["answers"]) if isinstance(ex["answers"], str) else ex["answers"]
        except Exception:
            continue
        names = [t["name"] for t in tools if "name" in t]
        if len(names) < 2 or not answers:
            continue
        first = answers[0].get("name")
        if first not in names:
            continue
        descs = {t["name"]: (t.get("description") or "")[:200] for t in tools if "name" in t}
        yield Record(f"xlam-{i}", "xlam_tools", "en", {"user_request": ex["query"]},
                     {"tool": choice_q("Which tool should be called first?", names, descs)}, {"tool": first},
                     split="eval" if i % 50 == 0 else "train")


# ----------------------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/decision")
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=int, default=0, help="per split cap (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in args.only.split(",") if n] or list(REGISTRY)
    manifest = {}
    for name in names:
        rng = random.Random(f"{args.seed}-{name}")
        print(f"== {name}", flush=True)
        try:
            recs = list(REGISTRY[name](args.limit, rng))
        except Exception:
            traceback.print_exc()
            manifest[name] = {"error": traceback.format_exc().splitlines()[-1]}
            continue
        counts = {}
        for split in ("train", "eval"):
            sub = [r for r in recs if r.split == split]
            if name in EVAL_ONLY and split == "train":
                sub = []
            if sub:
                n = write_jsonl(out / f"{name}.{split}.jsonl", sub)
                counts[split] = n
        manifest[name] = counts
        print(f"   {counts}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
