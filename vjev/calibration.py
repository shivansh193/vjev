"""Calibrators: temperature for option readouts, logistic maps for existence and boxes.

The shipped defaults were fitted on held-out splits for Qwen2.5-VL-3B-Instruct:
  * choice / noul temperatures: Oxford-IIIT Pets readout experiment
  * existence (Platt) and box confidence (logistic): RefCOCO readout experiment
They transfer imperfectly to other domains; refit with `fit_temperature` /
`fit_logistic` on a few hundred labelled examples from your own data.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.join(os.path.dirname(__file__), "calibration")


def softmax(z, t: float = 1.0):
    z = np.asarray(z, float) / t
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.asarray(x, float)))


def fit_temperature(logit_rows, correct_idx) -> float:
    """One temperature minimising NLL over option logits (never changes the argmax)."""
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 160))
    rows = [np.asarray(z, float) for z in logit_rows]
    nll = lambda t: -np.mean([np.log(max(softmax(z, t)[c], 1e-12)) for z, c in zip(rows, correct_idx)])
    return float(min(grid, key=nll))


@dataclass
class Logistic:
    """Standardise features, then a linear logit; serialisable."""
    mu: List[float]
    sd: List[float]
    w: List[float]

    def __call__(self, X) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        Z = np.c_[(X - np.array(self.mu)) / np.array(self.sd), np.ones(len(X))]
        return sigmoid(Z @ np.array(self.w))

    @classmethod
    def from_dict(cls, d: dict) -> "Logistic":
        return cls(list(d["mu"]), list(d["sd"]), list(d["w"]))


def fit_logistic(X, y, l2: float = 1e-2, iters: int = 4000, lr: float = 0.1) -> Logistic:
    X, y = np.atleast_2d(np.asarray(X, float)), np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = sigmoid(Z @ w)
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0])
    return Logistic(mu.tolist(), sd.tolist(), w.tolist())


PRESETS = {                       # mirror pass runs when first-pass P(box correct) is inside the band
    "fast": None,
    "balanced": (0.6, 0.9),
    "accurate": (0.0, 1.0),
}


@dataclass
class Calibration:
    choice_temperature: float = 0.767
    score_temperature: float = 0.767
    noul_platt: Logistic = field(default_factory=lambda: Logistic([0.0], [1.0], [1 / 0.601, 0.0]))
    existence_platt: Optional[Logistic] = None
    box_first: Optional[Logistic] = None
    box_full: Optional[Logistic] = None
    fills: Dict[str, float] = field(default_factory=dict)
    abstain_threshold: float = 0.6

    @classmethod
    def default(cls, name: str = "qwen2.5-vl-3b") -> "Calibration":
        cal = cls()
        path = os.path.join(HERE, f"{name}-box.json")
        if os.path.exists(path):
            d = json.load(open(path))
            cal.existence_platt = Logistic.from_dict(d["existence_platt"])
            cal.box_first = Logistic.from_dict(d["box_first_pass"])
            cal.box_full = Logistic.from_dict(d["box_full"])
            cal.fills = dict(d.get("fills", {}))
            cal.abstain_threshold = float(d.get("abstain_threshold", 0.6))
        return cal

    def fill(self, name: str, value: Optional[float]) -> float:
        """Replace a non-finite readout with the calibration-split median."""
        if value is None or not np.isfinite(value):
            return float(self.fills.get(name, 0.0))
        return float(value)
