"""Request / response types.

Field names deliberately mirror the TypeSafe `POST /v1/systemone` contract so that
code written against the TypeSafe SDK can be pointed at OpenThai-SystemOne unchanged:

    state      : str | dict | list        -- the thing to judge
    questions  : {id: Choice|Score|Noul}  -- typed questions
    answers    : {id: ChoiceAnswer|ScoreAnswer|NoulAnswer}
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_OPTIONS = 255  # single-stage cardinality limit (slots 0..254); slot 255 = abstain
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


class Noul(BaseModel):
    """A yes/no question. Returns p(yes)."""

    type: Literal["noul"] = "noul"
    instructions: str
    criteria: Optional[Dict[str, Optional[str]]] = None  # {"true": "...", "false": "..."}

    @field_validator("criteria")
    @classmethod
    def _check_criteria(cls, v):
        if v is None:
            return v
        extra = set(v) - {"true", "false"}
        if extra:
            raise ValueError(f"noul criteria keys must be 'true'/'false', got {sorted(extra)}")
        return v


class Choice(BaseModel):
    """Pick one option. `criteria` maps option name -> description (or null)."""

    type: Literal["choice"] = "choice"
    instructions: str
    criteria: Dict[str, Optional[str]]

    @field_validator("criteria")
    @classmethod
    def _check_criteria(cls, v):
        if len(v) < 1:
            raise ValueError("choice needs at least one option")
        if len(v) > MAX_OPTIONS:
            raise ValueError(f"choice supports at most {MAX_OPTIONS} options in one stage")
        for k in v:
            if not str(k).strip():
                raise ValueError("option names must be non-empty")
        return v


class Score(BaseModel):
    """Rate the state against ordered levels; `criteria[i]` describes level i (low -> high)."""

    type: Literal["score"] = "score"
    instructions: str
    criteria: List[str]

    @field_validator("criteria")
    @classmethod
    def _check_criteria(cls, v):
        if not (MIN_SCORE_LEVELS <= len(v) <= MAX_SCORE_LEVELS):
            raise ValueError(f"score needs {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels")
        return v


Question = Union[Noul, Choice, Score]


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: Dict[str, float]
    confidence: float
    abstain: Optional[float] = None  # OpenThai extension: p(none of the options); not in TypeSafe


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: Dict[int, str]
    probabilities: Dict[str, float]
    confidence: float


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int = 0
    permutations: int = 1  # OpenThai extension: number of option orders averaged (order-invariant mode)
    truncated: bool = False  # OpenThai extension: state was cut to max_state_tokens, the model did not see all of it


class SystemOneRequest(BaseModel):
    state: Union[str, Dict[str, Any], List[Any]]
    model: str = "openthai-systemone"
    questions: Dict[str, Question] = Field(discriminator=None)
    # OpenThai extensions. order_invariant=True averages the answer over several option orders (removes position
    # bias, ~2x latency); None = automatic (on for choice questions with > 10 options); permutations overrides the count.
    order_invariant: Optional[bool] = None
    permutations: Optional[int] = Field(default=None, ge=1, le=32)

    @model_validator(mode="after")
    def _non_empty(self):
        if not self.questions:
            raise ValueError("at least one question is required")
        return self


class SystemOneResponse(BaseModel):
    model: str
    answers: Dict[str, Answer]
    usage: Usage


def parse_question(obj: Union[Question, Dict[str, Any]]) -> Question:
    """Accept a dict (raw JSON) or an already-typed question."""
    if isinstance(obj, (Noul, Choice, Score)):
        return obj
    t = obj.get("type")
    if t == "noul":
        return Noul(**obj)
    if t == "choice":
        return Choice(**obj)
    if t == "score":
        return Score(**obj)
    raise ValueError(f"unknown question type: {t!r}")
