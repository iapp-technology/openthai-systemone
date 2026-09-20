from .types import Choice, Noul, Score, ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneRequest, SystemOneResponse
from .configuration import OpenThaiSystemOneConfig
from .modeling import OpenThaiSystemOneForDecision
from .formatting import Formatter, SPECIAL_TOKENS, N_SLOTS, ABSTAIN_SLOT
from .client import SystemOneClient

from transformers import AutoConfig, AutoModel

try:  # make AutoConfig/AutoModel aware of the architecture when the package is installed
    AutoConfig.register("openthai_systemone", OpenThaiSystemOneConfig)
    AutoModel.register(OpenThaiSystemOneConfig, OpenThaiSystemOneForDecision)
except ValueError:  # already registered (e.g. remote code + package both imported)
    pass

__version__ = "0.1.0"
__all__ = [
    "Choice", "Noul", "Score", "ChoiceAnswer", "NoulAnswer", "ScoreAnswer",
    "SystemOneRequest", "SystemOneResponse",
    "OpenThaiSystemOneConfig", "OpenThaiSystemOneForDecision",
    "Formatter", "SPECIAL_TOKENS", "N_SLOTS", "ABSTAIN_SLOT", "SystemOneClient",
]
