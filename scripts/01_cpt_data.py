"""Stage 1 data: stream the CPT mixture, tokenize, pack into fixed-length sequences, write uint32 shards.

    python scripts/01_cpt_data.py --tokenizer base/qwen3.5-0.8b-text --out data/cpt --tokens 5e9
    python scripts/01_cpt_data.py --tokenizer base/qwen3.5-0.8b-text --out data/cpt --tokens 2e6   # smoke

Mixture (by token share) -- edit MIXTURE to change:
    thai_web   0.55  HuggingFaceFW/fineweb-2  tha_Thai   (filtered CC Thai; Mangosteen-style quality filters applied here)
    thai_wiki  0.05  wikimedia/wikipedia 20231101.th
    thai_edu   0.05  (placeholder: Thai gov/edu/legal set; falls back to more fineweb-2 if missing)
    parallel   0.10  Thai<->EN pairs rendered as "th: ...\nen: ..." (Helsinki-NLP/opus-100 en-th, scb_mt_enth_2020)
    en_replay  0.15  HuggingFaceFW/fineweb-edu sample-10BT
    machine    0.10  HTML / accessibility trees / JSON / logs (bigcode/the-stack-smol subsets html+json, + synthetic UI states)

Each shard: <out>/shard-XXXXX.bin (uint32 token ids, contiguous, packed with EOS between docs) and a meta.json.
Output is deterministic per --seed so it can be regenerated on the H100 server.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MIXTURE = {
    "thai_web": 0.55,
    "thai_wiki": 0.05,
    "thai_edu": 0.05,
    "parallel": 0.10,
    "en_replay": 0.15,
    "machine": 0.10,
}

THAI_RE = re.compile(r"[฀-๿]")


def thai_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(THAI_RE.findall(text)) / max(1, len(text))


def quality_ok_thai(text: str) -> bool:
    """Cheap Mangosteen-style filters: enough Thai, not too short, low boilerplate/duplication."""
    if len(text) < 200 or thai_ratio(text) < 0.5:
        return False
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    uniq = len(set(lines)) / len(lines)
    if uniq < 0.7:
        return False
    if text.count("http") > len(text) / 200:
        return False
    bad = ("คลิกที่นี่", "สมัครสมาชิก", "เว็บพนัน", "บาคาร่า", "สล็อต", "หวยออนไลน์")
    if sum(text.count(b) for b in bad) > 3:
        return False
    return True


def stream(name, config=None, split="train", **kw):
    from datasets import load_dataset

    return load_dataset(name, config, split=split, streaming=True, **kw)


def src_thai_web(rng):
    for ex in stream("HuggingFaceFW/fineweb-2", "tha_Thai"):
        t = ex.get("text", "")
        if quality_ok_thai(t):
            yield t


def src_thai_wiki(rng):
    for ex in stream("wikimedia/wikipedia", "20231101.th"):
        t = ex.get("text", "")
        if len(t) > 300:
            yield f"{ex.get('title', '')}\n\n{t}"


def src_thai_edu(rng):
    # Best-effort: a few open Thai instruction/edu corpora rendered as plain text; falls back to web if unavailable.
    tried = 0
    for name, cfg, render in [
        ("airesearch/WangchanThaiInstruct", None, lambda e: f"{e.get('Instruction', '')}\n{e.get('Input', '') or ''}\n{e.get('Output', '')}"),
        ("pythainlp/thaisum", None, lambda e: f"{e.get('title', '')}\n{e.get('body', '')}"),
    ]:
        try:
            for ex in stream(name, cfg):
                t = render(ex)
                if len(t) > 200:
                    yield t
            tried += 1
        except Exception as e:  # dataset missing / renamed
            print(f"[thai_edu] skip {name}: {e}", file=sys.stderr)
    if tried == 0:
        yield from src_thai_web(rng)


def src_parallel(rng):
    for ex in stream("Helsinki-NLP/opus-100", "en-th"):
        tr = ex.get("translation", {})
        if tr.get("th") and tr.get("en"):
            if rng.random() < 0.5:
                yield f"th: {tr['th']}\nen: {tr['en']}"
            else:
                yield f"en: {tr['en']}\nth: {tr['th']}"


def src_en_replay(rng):
    for ex in stream("HuggingFaceFW/fineweb-edu", "sample-10BT"):
        t = ex.get("text", "")
        if len(t) > 300:
            yield t


def _synthetic_ui_state(rng) -> str:
    """Cheap procedurally generated accessibility-tree-like text so the LM sees the genre during CPT."""
    th_words = ["ชำระเงิน", "สมัครสมาชิก", "เข้าสู่ระบบ", "ค้นหา", "ตะกร้าสินค้า", "ที่อยู่จัดส่ง", "อีเมล", "รหัสผ่าน", "ยืนยัน", "ยกเลิก",
                "เบอร์โทรศัพท์", "จังหวัด", "อำเภอ", "รหัสไปรษณีย์", "บันทึก", "ถัดไป", "ย้อนกลับ", "ดาวน์โหลด", "แชร์", "ตั้งค่า"]
    en_words = ["Submit", "Cancel", "Search", "Login", "Sign up", "Cart", "Checkout", "Email", "Password", "Next", "Back", "Settings"]
    roles = ["button", "link", "textbox", "checkbox", "combobox", "menuitem", "heading", "tab", "radio", "img"]
    lines = [f"url: https://{rng.choice(['shop', 'bank', 'gov', 'app', 'portal'])}.example.{rng.choice(['co.th', 'com', 'go.th'])}/{rng.choice(['checkout', 'login', 'profile', 'search', 'orders'])}"]
    for i in range(rng.randint(8, 40)):
        role = rng.choice(roles)
        name = rng.choice(th_words if rng.random() < 0.7 else en_words)
        extra = ""
        if role in ("textbox", "combobox"):
            extra = f' value="{rng.choice(["", "somchai@example.com", "081-234-5678", "10110"])}"'
        if role == "checkbox":
            extra = f" checked={str(rng.random() < 0.5).lower()}"
        lines.append(f'[{i}] {role} "{name}"{extra}')
    return "\n".join(lines)


def src_machine(rng):
    it = None
    try:
        it = iter(stream("bigcode/the-stack-smol", data_dir="data/html"))
    except Exception as e:
        print(f"[machine] the-stack-smol html unavailable: {e}", file=sys.stderr)
    while True:
        r = rng.random()
        if r < 0.4:
            yield _synthetic_ui_state(rng)
        elif r < 0.7:
            # random JSON records
            rec = {"order_id": rng.randint(10000, 99999), "customer": rng.choice(["สมชาย ใจดี", "Somsri", "นภา วงศ์"]),
                   "items": [{"sku": f"SKU{rng.randint(100, 999)}", "qty": rng.randint(1, 5), "price": rng.randint(50, 5000)} for _ in range(rng.randint(1, 4))],
                   "status": rng.choice(["paid", "pending", "shipped", "cancelled", "ยกเลิก", "จัดส่งแล้ว"]),
                   "note": rng.choice(["", "ส่งด่วน", "โทรก่อนส่ง", "leave at door"])}
            yield json.dumps(rec, ensure_ascii=False, indent=rng.choice([None, 2]))
        elif it is not None:
            try:
                ex = next(it)
                t = ex.get("content", "")
                if 200 < len(t) < 20000:
                    yield t
            except StopIteration:
                it = None
        else:
            yield _synthetic_ui_state(rng)


SOURCES = {
    "thai_web": src_thai_web,
    "thai_wiki": src_thai_wiki,
    "thai_edu": src_thai_edu,
    "parallel": src_parallel,
    "en_replay": src_en_replay,
    "machine": src_machine,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="base/qwen3.5-0.8b-text")
    ap.add_argument("--out", default="data/cpt")
    ap.add_argument("--tokens", type=float, default=5e9)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-tokens", type=int, default=2_000_000)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tok.eos_token_id
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    iters = {k: SOURCES[k](random.Random(f"{args.seed}-{k}")) for k in MIXTURE}
    budget = {k: int(args.tokens * w) for k, w in MIXTURE.items()}
    got = {k: 0 for k in MIXTURE}
    docs = {k: 0 for k in MIXTURE}

    buf = np.zeros(args.shard_tokens, dtype=np.uint32)
    n = 0
    shard = 0
    total = 0
    eval_buf = []
    eval_n = 0

    def flush():
        nonlocal n, shard
        if n == 0:
            return
        buf[:n].tofile(out / f"shard-{shard:05d}.bin")
        shard += 1
        n = 0

    while any(got[k] < budget[k] for k in MIXTURE):
        remaining = [k for k in MIXTURE if got[k] < budget[k]]
        k = rng.choices(remaining, weights=[MIXTURE[x] for x in remaining])[0]
        try:
            text = next(iters[k])
        except StopIteration:
            print(f"[{k}] exhausted at {got[k]} tokens", file=sys.stderr)
            budget[k] = got[k]
            continue
        ids = tok(text, add_special_tokens=False)["input_ids"] + [eos]
        got[k] += len(ids)
        docs[k] += 1
        if eval_n < args.eval_tokens and rng.random() < 0.01:
            eval_buf.extend(ids)
            eval_n += len(ids)
            continue
        i = 0
        while i < len(ids):
            take = min(len(ids) - i, args.shard_tokens - n)
            buf[n : n + take] = ids[i : i + take]
            n += take
            i += take
            total += take
            if n == args.shard_tokens:
                flush()
        if total and total % 50_000_000 < len(ids):
            print(f"tokens={total / 1e6:.0f}M " + " ".join(f"{k}={got[k] / 1e6:.0f}M" for k in MIXTURE), flush=True)
    flush()
    np.asarray(eval_buf, dtype=np.uint32).tofile(out / "eval.bin")
    meta = {"tokens": int(total), "eval_tokens": int(eval_n), "shards": shard, "by_source_tokens": got, "by_source_docs": docs,
            "mixture": MIXTURE, "tokenizer": args.tokenizer, "eos": eos, "seed": args.seed}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
