"""End-to-end: tiny random model -> SystemOneClient -> FastAPI contract. CPU only."""
import json

import pytest
import torch
from transformers import AutoTokenizer

from openthai_systemone.client import SystemOneClient
from openthai_systemone.formatting import add_special_tokens
from openthai_systemone.modeling import OpenThaiSystemOneForDecision
from openthai_systemone.types import Choice, Noul, Score

from tests.test_modeling import tiny_config


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny-typesafe")
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    add_special_tokens(tok)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    cfg = tiny_config(vocab=len(tok))
    cfg.answer_token_id = tok.convert_tokens_to_ids("<|ts_answer|>")
    cfg.pad_token_id = tok.pad_token_id
    torch.manual_seed(0)
    m = OpenThaiSystemOneForDecision(cfg)
    m.save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


def test_client_contract(model_dir):
    c = SystemOneClient(model_dir, device="cpu")
    resp = c.system_one(
        {"ticket": "ขอเงินคืนด้วยครับ ระบบล่มตั้งแต่เมื่อวาน"},
        {
            "department": Choice(instructions="Which team?", criteria={"billing": "money", "technical": None, "sales": None}),
            "frustration": Score(instructions="How angry?", criteria=["calm", "civil", "furious"]),
            "refund": Noul(instructions="Refund requested?"),
        },
    )
    d = json.loads(resp.model_dump_json())
    assert set(d) == {"model", "answers", "usage"}
    dep = d["answers"]["department"]
    assert dep["type"] == "choice" and dep["choice"] in dep["probabilities"]
    assert abs(sum(dep["probabilities"].values()) - 1) < 1e-5 and 0 <= dep["confidence"] <= 1
    fr = d["answers"]["frustration"]
    assert fr["type"] == "score" and 0 <= fr["score"] <= 2 and fr["legend"]["0"] == "calm"
    assert 0 <= d["answers"]["refund"]["noul"] <= 1
    assert d["usage"]["input_tokens"] > 0


def test_batch_matches_single(model_dir):
    c = SystemOneClient(model_dir, device="cpu")
    qs = {"q": Choice(instructions="?", criteria={"a": None, "b": None, "c": None, "d": None})}
    single = [c.system_one(s, qs).answers["q"].probabilities for s in ("short", "a much longer state text " * 5)]
    batch = [r.answers["q"].probabilities for r in c.system_one_batch([("short", qs), ("a much longer state text " * 5, qs)])]
    for s, b in zip(single, batch):
        for k in s:
            assert abs(s[k] - b[k]) < 1e-4


def test_usage_reports_truncation(model_dir):
    c = SystemOneClient(model_dir, device="cpu", max_state_tokens=32)
    qs = {"q": Noul(instructions="ok?")}
    short = c.system_one("hello", qs)
    long = c.system_one("a long state that will not fit " * 20, qs)
    assert short.usage.truncated is False
    assert long.usage.truncated is True
    assert long.usage.input_tokens < short.usage.input_tokens + 32
    assert json.loads(long.model_dump_json())["usage"]["truncated"] is True


def test_http_server(model_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from openthai_systemone import server

    monkeypatch.setenv("OPENTHAI_SYSTEMONE_MODEL", model_dir)
    server.get_client.cache_clear()
    with TestClient(server.app) as tc:
        r = tc.post("/v1/systemone", json={
            "state": "hello",
            "model": "jev-latest",
            "questions": {
                "x": {"type": "choice", "instructions": "?", "criteria": {"yes": None, "no": None}},
                "n": {"type": "noul", "instructions": "ok?"},
            },
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["model"] == "jev-latest" and set(body["answers"]) == {"x", "n"}
        bad = tc.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "score", "instructions": "?", "criteria": ["only-one"]}}})
        assert bad.status_code == 422


def test_permutation_averaging_is_order_invariant(model_dir):
    c = SystemOneClient(model_dir, device="cpu")
    a = {"x": "first", "y": "second"}
    b = {"y": "second", "x": "first"}
    ra = c.system_one("some state", {"q": Choice(instructions="?", criteria=a)}, permutations=2)
    rb = c.system_one("some state", {"q": Choice(instructions="?", criteria=b)}, permutations=2)
    # with k=2 the two cyclic shifts cover both orders, so both calls average the same two encodings
    for k in a:
        assert abs(ra.answers["q"].probabilities[k] - rb.answers["q"].probabilities[k]) < 1e-5
    assert ra.usage.permutations == 2
    single = c.system_one("some state", {"q": Choice(instructions="?", criteria=a)}, permutations=1)
    assert single.usage.permutations == 1


def test_auto_mode_turns_on_for_large_option_sets(model_dir):
    c = SystemOneClient(model_dir, device="cpu")
    small = c.system_one("s", {"q": Choice(instructions="?", criteria={f"o{i}": None for i in range(5)})})
    large = c.system_one("s", {"q": Choice(instructions="?", criteria={f"o{i}": None for i in range(20)})})
    assert small.usage.permutations == 1 and large.usage.permutations == 8
    assert abs(sum(large.answers["q"].probabilities.values()) - 1) < 1e-5 and len(large.answers["q"].probabilities) == 20
    # score / noul questions are never permuted and still decode correctly alongside a permuted choice
    mixed = c.system_one("s", {"q": Choice(instructions="?", criteria={f"o{i}": None for i in range(20)}),
                               "lvl": Score(instructions="?", criteria=["a", "b", "c"]), "yn": Noul(instructions="?")}, permutations=3)
    assert mixed.usage.permutations == 3 and 0 <= mixed.answers["lvl"].score <= 2 and 0 <= mixed.answers["yn"].noul <= 1


def test_server_order_invariant_flag(model_dir, monkeypatch):
    from fastapi.testclient import TestClient

    from openthai_systemone import server

    monkeypatch.setenv("OPENTHAI_SYSTEMONE_MODEL", model_dir)
    server.get_client.cache_clear()
    with TestClient(server.app) as tc:
        body = {"state": "x", "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {"a": None, "b": None, "c": None}}}}
        assert tc.post("/v1/systemone", json=body).json()["usage"]["permutations"] == 1
        assert tc.post("/v1/systemone", json={**body, "order_invariant": True}).json()["usage"]["permutations"] == 3
        assert tc.post("/v1/systemone", json={**body, "permutations": 2}).json()["usage"]["permutations"] == 2
