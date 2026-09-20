"""Unified decision record format shared by all data builders and the SFT trainer.

    {
      "id": str, "source": str, "lang": "th"|"en"|"mixed"|..., "split": "train"|"eval",
      "state": str | dict | list,
      "questions": {qid: {"type": "choice"|"score"|"noul", "instructions": str, "criteria": ...}},
      "labels":    {qid: option-name | level-index | bool}
    }
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

# Instruction paraphrases: the model must not overfit to one phrasing per dataset.
TH_CHOICE_TEMPLATES = [
    "ข้อความนี้จัดอยู่ในหมวด{what}ใด",
    "เลือก{what}ที่ตรงกับข้อความมากที่สุด",
    "จาก state ด้านบน {what}ที่ถูกต้องคือข้อใด",
    "ระบุ{what}ของข้อความนี้",
]
EN_CHOICE_TEMPLATES = [
    "Which {what} best matches this text?",
    "Classify the {what} of the state above.",
    "Select the correct {what}.",
    "What is the {what}?",
]
TH_NOUL_TEMPLATES = ["ข้อความนี้{what}หรือไม่", "จริงหรือไม่ว่า ข้อความนี้{what}", "state นี้{what}ใช่หรือไม่"]
EN_NOUL_TEMPLATES = ["Is this text {what}?", "Does the state {what}?", "True or false: the text {what}."]


@dataclass
class Record:
    id: str
    source: str
    lang: str
    state: Union[str, Dict[str, Any], List[Any]]
    questions: Dict[str, Dict[str, Any]]
    labels: Dict[str, Union[str, int, bool]]
    split: str = "train"
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "Record":
        d = json.loads(line)
        known = {"id", "source", "lang", "state", "questions", "labels", "split", "meta"}
        meta = dict(d.get("meta", {}))
        meta.update({k: v for k, v in d.items() if k not in known})
        return Record(**{k: d[k] for k in ("id", "source", "lang", "state", "questions", "labels")},
                      split=d.get("split", "train"), meta=meta)


def choice_q(instructions: str, options: Iterable[str], descs: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Any]:
    descs = descs or {}
    return {"type": "choice", "instructions": instructions, "criteria": {o: descs.get(o) for o in options}}


def score_q(instructions: str, levels: List[str]) -> Dict[str, Any]:
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def noul_q(instructions: str, true_desc: Optional[str] = None, false_desc: Optional[str] = None) -> Dict[str, Any]:
    q: Dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true_desc or false_desc:
        q["criteria"] = {"true": true_desc, "false": false_desc}
    return q


def pick_template(rng: random.Random, lang: str, kind: str, what: str) -> str:
    if kind == "choice":
        t = TH_CHOICE_TEMPLATES if lang == "th" else EN_CHOICE_TEMPLATES
    else:
        t = TH_NOUL_TEMPLATES if lang == "th" else EN_NOUL_TEMPLATES
    return rng.choice(t).format(what=what)


def write_jsonl(path, records: Iterable[Record]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path) -> Iterator[Record]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Record.from_json(line)
