from __future__ import annotations

from typing import Any, Dict, Optional

from transformers import AutoConfig, PretrainedConfig


class OpenThaiSystemOneConfig(PretrainedConfig):
    """Config = a text-tower config (Qwen3.5 text by default) + slot-head settings."""

    model_type = "openthai_systemone"
    sub_configs = {"text_config": AutoConfig}

    def __init__(
        self,
        text_config: Optional[Dict[str, Any] | PretrainedConfig] = None,
        n_slots: int = 256,
        abstain_slot: int = 255,
        answer_token_id: Optional[int] = None,
        head_bias: bool = True,
        n_temperatures: int = 3,  # per question type: choice / score / noul
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = AutoConfig.for_model(**text_config) if "model_type" in text_config else AutoConfig.for_model("qwen3_5_text", **text_config)
        self.text_config = text_config
        self.n_slots = n_slots
        self.abstain_slot = abstain_slot
        self.answer_token_id = answer_token_id
        self.head_bias = head_bias
        self.n_temperatures = n_temperatures
        super().__init__(**kwargs)

    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size
