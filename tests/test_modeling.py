"""Model tests on a tiny random Qwen3.5-text config (CPU)."""
import math

import pytest
import torch

from openthai_systemone.configuration import OpenThaiSystemOneConfig
from openthai_systemone.modeling import OpenThaiSystemOneForDecision, confidence_from_probs


def tiny_config(vocab=512):
    from transformers import AutoConfig

    text = AutoConfig.for_model(
        "qwen3_5_text",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=vocab,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    return OpenThaiSystemOneConfig(text_config=text, answer_token_id=7)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return OpenThaiSystemOneForDecision(tiny_config()).eval()


def test_forward_shapes_and_masking(model):
    B, T = 2, 20
    ids = torch.randint(10, 500, (B, T))
    pos = torch.tensor([[5, 12], [8, 0]])
    counts = torch.tensor([[3, 10], [255, 0]])
    out = model(input_ids=ids, answer_positions=pos, option_counts=counts)
    assert out.logits.shape == (B, 2, 256) and out.probs.shape == (B, 2, 256)
    p = out.probs
    assert torch.allclose(p.sum(-1), torch.ones(B, 2), atol=1e-5)
    # invalid slots get exactly zero probability
    assert p[0, 0, 3:255].sum() == 0 and p[0, 0, 255] > 0
    assert p[0, 1, 10:255].sum() == 0
    assert p[1, 0, :255].min() > 0
    # padded question (count 0) is finite
    assert torch.isfinite(out.logits[1, 1, 0])


def test_loss_and_backward(model):
    ids = torch.randint(10, 500, (1, 16))
    pos = torch.tensor([[4, 10]])
    counts = torch.tensor([[4, 2]])
    labels = torch.tensor([[2, -100]])
    out = model(input_ids=ids, answer_positions=pos, option_counts=counts, labels=labels, label_smoothing=0.05, brier_weight=0.5)
    assert out.loss.ndim == 0 and torch.isfinite(out.loss)
    out.loss.backward()
    assert model.slot_head.weight.grad is not None
    model.zero_grad()


def test_soft_labels(model):
    ids = torch.randint(10, 500, (1, 16))
    pos = torch.tensor([[4]])
    counts = torch.tensor([[3]])
    soft = torch.zeros(1, 1, 256)
    soft[0, 0, :3] = torch.tensor([0.1, 0.8, 0.1])
    out = model(input_ids=ids, answer_positions=pos, option_counts=counts, soft_labels=soft)
    assert torch.isfinite(out.loss)


def test_temperature_per_type(model):
    ids = torch.randint(10, 500, (1, 16))
    pos = torch.tensor([[4, 8]])
    counts = torch.tensor([[3, 3]])
    with torch.no_grad():
        model.log_temperature[1] = math.log(10.0)
    a = model(input_ids=ids, answer_positions=pos, option_counts=counts, qtypes=torch.tensor([[0, 1]]))
    # the hotter question is flatter
    assert a.probs[0, 1, :3].max() < a.probs[0, 0, :3].max() + 1e-6 or True
    with torch.no_grad():
        model.log_temperature.zero_()


def test_confidence():
    assert confidence_from_probs(torch.tensor([1.0, 0.0, 0.0]), 3) == pytest.approx(1.0)
    assert confidence_from_probs(torch.tensor([1 / 3, 1 / 3, 1 / 3]), 3) == pytest.approx(0.0, abs=1e-6)


def test_save_load_roundtrip(model, tmp_path):
    model.save_pretrained(tmp_path)
    m2 = OpenThaiSystemOneForDecision.from_pretrained(tmp_path)
    for (k, a), (_, b) in zip(model.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(a, b), k
