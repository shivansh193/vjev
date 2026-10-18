"""Typed, calibrated decisions about images, read from a VLM's token probabilities.

    from vjev import Decider
    d = Decider()                                   # Qwen2.5-VL-3B, preset "balanced"
    out = d.decide({"assets": {"img": "photo.jpg"},
                    "questions": {"dog": {"type": "box", "target": "the dog", "image": "img"}}})
    out["answers"]["dog"]   # {"type": "box", "abstained": False, "box": {...}, "confidence": 0.91, ...}

No answer is generated free-form: choice / score / noul answers are softmaxes over the
option tokens at the answer position; a box is the model's own short grounding output with
a calibrated confidence, gated by an existence check so it can abstain.
"""
import json
import re
import time
from typing import Optional

import numpy as np

from .backend import load_image
from .calibration import PRESETS, Calibration, softmax
from .schema import Box, Choice, Noul, Request, Score, parse_request

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def swap_left_right(text: str) -> str:
    """Mirroring an image swaps which object 'left' / 'right' refers to."""
    return re.sub(r"\b(left|right)\b", lambda m: {"left": "right", "right": "left"}[m.group(1).lower()], text,
                  flags=re.IGNORECASE)


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class Decider:
    def __init__(self, backend=None, calibration: Optional[Calibration] = None, preset: str = "balanced"):
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {sorted(PRESETS)}")
        if backend is None:
            from .backend import QwenVLBackend
            backend = QwenVLBackend()
        self.backend = backend
        self.cal = calibration or Calibration.default()
        self.preset = preset

    # ------------------------------------------------------------ prompts

    @staticmethod
    def _context(state) -> str:
        return f"Context: {json.dumps(state, ensure_ascii=False)}\n" if state else ""

    # ------------------------------------------------------------ question types

    def choice(self, q: Choice, image, state=None) -> dict:
        k = len(q.options)
        letters = list(LETTERS[:k])

        def read(order):
            text = (self._context(state) + f"{q.question}\nOptions:\n"
                    + "\n".join(f"{letters[i]}. {q.options[o]}" for i, o in enumerate(order))
                    + "\nAnswer with the letter only.")
            return self.backend.next_token_logits(image, text, letters)

        if q.rotate:
            logp = np.zeros(k)
            for r in range(k):
                order = list(range(r, k)) + list(range(r))
                lp = np.log(np.clip(softmax(read(order)), 1e-12, 1))
                for pos, o in enumerate(order):
                    logp[o] += lp[pos] / k
            logits = logp
        else:
            logits = read(list(range(k)))
        p = softmax(logits, self.cal.choice_temperature)
        best = int(p.argmax())
        return {"type": "choice", "choice": q.options[best], "confidence": round(float(p[best]), 4),
                "probabilities": {o: round(float(v), 4) for o, v in zip(q.options, p)}}

    def score(self, q: Score, image, state=None) -> dict:
        digits = [str(i + 1) for i in range(len(q.levels))]
        text = (self._context(state) + f"{q.question}\nScale (lowest to highest):\n"
                + "\n".join(f"{d}. {lv}" for d, lv in zip(digits, q.levels))
                + "\nAnswer with the number only.")
        p = softmax(self.backend.next_token_logits(image, text, digits), self.cal.score_temperature)
        best = int(p.argmax())
        return {"type": "score", "level": q.levels[best], "index": best,
                "expected_index": round(float((p * np.arange(len(p))).sum()), 4),
                "confidence": round(float(p[best]), 4),
                "distribution": {lv: round(float(v), 4) for lv, v in zip(q.levels, p)}}

    def _yes_no_logit(self, image, text) -> float:
        z = self.backend.next_token_logits(image, text + " Answer Yes or No.", ["Yes", "No"])
        return float(z[0] - z[1])

    def noul(self, q: Noul, image, state=None) -> dict:
        z = self._yes_no_logit(image, self._context(state) + q.question)
        p = float(self.cal.noul_platt([[z]])[0])
        return {"type": "noul", "answer": p >= 0.5, "probability": round(p, 4)}

    def box(self, q: Box, image, state=None) -> dict:
        from PIL import ImageOps
        cal = self.cal
        if cal.existence_platt is None or cal.box_first is None:
            raise RuntimeError("box calibration not loaded")
        z = self._yes_no_logit(image, self._context(state) + f"Is there {q.target} in the image?")
        p_exist = float(cal.existence_platt([[cal.fill("noul_z", z)]])[0])
        if p_exist < cal.abstain_threshold:
            return {"type": "box", "abstained": True, "box": None, "exists": round(p_exist, 4),
                    "confidence": round(1 - p_exist, 4), "passes": 0}
        b, st = self.backend.ground(image, q.target)
        # feature names follow the calibration file (mean token probability was logged as "box_conf")
        first = [cal.fill("box_conf", st.get("mean_p")), cal.fill("min_p", st.get("min_p")),
                 cal.fill("mean_entropy", st.get("mean_entropy"))]
        p_box = float(cal.box_first([first])[0])
        passes = 1
        band = PRESETS[self.preset]
        if b is not None and band is not None and band[0] <= p_box <= band[1]:
            bm, _ = self.backend.ground(ImageOps.mirror(image), swap_left_right(q.target))
            agree = box_iou(b, [1 - bm[2], bm[1], 1 - bm[0], bm[3]]) if bm is not None else 0.0
            p_box = float(cal.box_full([first + [agree]])[0])
            passes = 2
        if b is None:                                  # model produced no parsable box
            return {"type": "box", "abstained": True, "box": None, "exists": round(p_exist, 4),
                    "confidence": round(1 - p_exist, 4), "passes": passes}
        return {"type": "box", "abstained": False, "exists": round(p_exist, 4),
                "box": {"left": round(b[0], 4), "top": round(b[1], 4), "right": round(b[2], 4), "bottom": round(b[3], 4)},
                "confidence": round(p_exist * p_box, 4), "box_confidence": round(p_box, 4), "passes": passes}

    # ------------------------------------------------------------ request

    def decide(self, request) -> dict:
        req = request if isinstance(request, Request) else parse_request(request)
        images = {name: load_image(a) for name, a in req.assets.items()}
        handlers = {Choice: self.choice, Score: self.score, Noul: self.noul, Box: self.box}
        answers, t0 = {}, time.time()
        for name, q in req.questions.items():
            answers[name] = handlers[type(q)](q, images.get(q.image), req.state)
        return {"answers": answers, "preset": self.preset, "seconds": round(time.time() - t0, 3)}
