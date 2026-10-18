"""Request / answer types for typed visual decisions.

A request is a state, a set of image assets and a set of named, typed questions:

    {
      "state": {...},                                  # optional JSON context
      "assets": {"shelf": <PIL image | path | base64>},
      "questions": {
        "breed":   {"type": "choice", "question": "Which breed?", "options": ["beagle", "pug"], "image": "shelf"},
        "fresh":   {"type": "score",  "question": "How fresh is it?", "levels": ["stale", "ok", "fresh"], "image": "shelf"},
        "has_dog": {"type": "noul",   "question": "Is there a dog?", "image": "shelf"},
        "dog":     {"type": "box",    "target": "the dog on the left", "image": "shelf"}
      }
    }

Every answer is bounded by its type: a choice is always one of the given options, a score
one of the given levels, a box either a box or an explicit abstention.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

MAX_CHOICE_OPTIONS = 26          # one letter token per option (A-Z)
MAX_SCORE_LEVELS = 9             # one digit token per level (1-9)


class RequestError(ValueError):
    """The request does not match the schema."""


@dataclass
class Choice:
    question: str
    options: List[str]
    image: Optional[str] = None
    rotate: bool = False         # average over cyclic option orders (removes position bias, K x cost)


@dataclass
class Score:
    question: str
    levels: List[str]            # ordered from lowest to highest
    image: Optional[str] = None


@dataclass
class Noul:
    question: str                # a yes / no question
    image: Optional[str] = None


@dataclass
class Box:
    target: str                  # referring expression, e.g. "the red mug behind the laptop"
    image: Optional[str] = None


Question = Union[Choice, Score, Noul, Box]


@dataclass
class Request:
    questions: Dict[str, Question]
    assets: Dict[str, object] = field(default_factory=dict)
    state: Optional[dict] = None


def _need(spec: dict, key: str, name: str):
    if key not in spec or spec[key] in (None, "", []):
        raise RequestError(f"question '{name}': missing '{key}'")
    return spec[key]


def parse_question(name: str, spec: dict) -> Question:
    kind = spec.get("type")
    image = spec.get("image")
    if kind == "choice":
        options = list(_need(spec, "options", name))
        if not 2 <= len(options) <= MAX_CHOICE_OPTIONS:
            raise RequestError(f"question '{name}': choice needs 2-{MAX_CHOICE_OPTIONS} options, got {len(options)}")
        if len(set(options)) != len(options):
            raise RequestError(f"question '{name}': options must be unique")
        return Choice(_need(spec, "question", name), [str(o) for o in options], image, bool(spec.get("rotate", False)))
    if kind == "score":
        levels = list(_need(spec, "levels", name))
        if not 2 <= len(levels) <= MAX_SCORE_LEVELS:
            raise RequestError(f"question '{name}': score needs 2-{MAX_SCORE_LEVELS} levels, got {len(levels)}")
        return Score(_need(spec, "question", name), [str(v) for v in levels], image)
    if kind == "noul":
        return Noul(_need(spec, "question", name), image)
    if kind == "box":
        if image is None:
            raise RequestError(f"question '{name}': box needs an image")
        return Box(_need(spec, "target", name), image)
    raise RequestError(f"question '{name}': unknown type {kind!r} (choice, score, noul, box)")


def parse_request(body: dict) -> Request:
    if not isinstance(body, dict):
        raise RequestError("request must be a JSON object")
    qs = body.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise RequestError("request needs a non-empty 'questions' object")
    questions = {name: parse_question(name, spec) for name, spec in qs.items()}
    assets = body.get("assets") or {}
    for name, q in questions.items():
        if q.image is not None and q.image not in assets:
            raise RequestError(f"question '{name}': image '{q.image}' is not in assets")
    return Request(questions, assets, body.get("state"))
