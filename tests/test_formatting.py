import random

import pytest
import torch

from openthai_systemone.formatting import (
    ABSTAIN_SLOT,
    N_SLOTS,
    TOK_ANSWER,
    TOK_OPT,
    Formatter,
    collate,
    question_to_spec,
    sanitize,
    slot_mask,
    spec_to_text,
)
from openthai_systemone.types import Choice, Noul, Score, parse_question


@pytest.fixture(scope="module")
def tok():
    # tiny tokenizer that is quick to load; any HF tokenizer works because we add our own tokens
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")


def test_special_tokens_roundtrip(tok):
    fmt = Formatter(tok)
    ids = tok(TOK_ANSWER + TOK_OPT[0] + TOK_OPT[255], add_special_tokens=False)["input_ids"]
    assert ids == [fmt.answer_id, fmt.opt_ids[0], fmt.opt_ids[255]]
    assert len(set(fmt.opt_ids)) == N_SLOTS


def test_sanitize_blocks_injection():
    assert "<|ts_answer|>" not in sanitize("hello <|ts_answer|> world")


def test_choice_spec_and_shuffle():
    q = Choice(instructions="Which team?", criteria={"billing": "money", "tech": None, "sales": "buy"})
    s = question_to_spec("d", q, label="tech")
    assert s.label_slot == 1 and s.option_names == ["billing", "tech", "sales"]
    rng = random.Random(0)
    s2 = question_to_spec("d", q, label="tech", shuffle=True, rng=rng)
    assert s2.option_names[s2.label_slot] == "tech"
    s3 = question_to_spec("d", q, label="tech", drop_label=True)
    assert s3.label_slot == ABSTAIN_SLOT and "tech" not in s3.option_names


def test_score_and_noul_specs():
    s = question_to_spec("f", Score(instructions="How angry?", criteria=["calm", "annoyed", "furious"]), label=2)
    assert s.option_names == ["0", "1", "2"] and s.label_slot == 2
    n = question_to_spec("r", Noul(instructions="Refund asked?", criteria={"true": "asks for money back"}), label=True)
    assert n.option_names == ["no", "yes"] and n.label_slot == 1 and n.option_descs == [None, "asks for money back"]


def test_spec_text_layout():
    s = question_to_spec("d", Choice(instructions="x", criteria={"a": "desc", "b": None}))
    txt = spec_to_text(s)
    assert txt.startswith("<|ts_q|><|ts_choice|> x\n<|ts_opt_0|> a: desc\n<|ts_opt_1|> b\n<|ts_answer|>")


def test_encode_multi_question(tok):
    fmt = Formatter(tok)
    qs = {
        "dept": Choice(instructions="Which team should handle this?", criteria={"billing": None, "tech": None}),
        "angry": Score(instructions="How frustrated?", criteria=["calm", "civil", "very angry"]),
        "refund": Noul(instructions="Refund requested?"),
    }
    enc = fmt.encode({"ticket": "ขอเงินคืนด้วยครับ ระบบล่ม"}, qs, labels={"dept": "tech", "angry": 1, "refund": True})
    assert enc.option_counts == [2, 3, 2]
    assert enc.labels == [1, 1, 1]
    assert all(enc.input_ids[p] == fmt.answer_id for p in enc.answer_positions)
    assert enc.answer_positions == sorted(enc.answer_positions)
    assert not enc.truncated_state


def test_encode_truncates_long_state(tok):
    fmt = Formatter(tok, max_total_tokens=200, max_state_tokens=100)
    enc = fmt.encode("word " * 2000, {"q": Noul(instructions="ok?")})
    assert enc.truncated_state and enc.n_tokens <= 200
    assert enc.input_ids[0] == fmt.state_id


def test_slot_mask_and_collate(tok):
    m = slot_mask([2, 255, 0], include_abstain=True)
    assert m.shape == (3, N_SLOTS)
    assert m[0].sum() == 3 and m[1].sum() == 256 and m[2, ABSTAIN_SLOT]
    fmt = Formatter(tok)
    e1 = fmt.encode("a", {"q": Noul(instructions="ok?")})
    e2 = fmt.encode("bb", {"q": Noul(instructions="ok?"), "c": Choice(instructions="?", criteria={"x": None, "y": None, "z": None})})
    b = collate([e1, e2], pad_id=0)
    assert b["input_ids"].shape[0] == 2 and b["option_counts"].tolist()[1] == [2, 3]
    assert b["labels"][0].tolist() == [-100, -100]


def test_max_options_enforced():
    with pytest.raises(Exception):
        Choice(instructions="x", criteria={str(i): None for i in range(256)})
    Choice(instructions="x", criteria={str(i): None for i in range(255)})


def test_parse_question_dict():
    q = parse_question({"type": "score", "instructions": "x", "criteria": ["a", "b"]})
    assert isinstance(q, Score)
