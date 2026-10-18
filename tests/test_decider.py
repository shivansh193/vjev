"""Decision logic tests with a fake backend (no GPU, no model weights)."""
import numpy as np
import pytest
from PIL import Image

from vjev import Calibration, Decider, RequestError, fit_logistic, fit_temperature, parse_request
from vjev.calibration import Logistic, softmax
from vjev.decider import box_iou, swap_left_right


class FakeBackend:
    """Logits favour a configured token; ground() returns a fixed box and token stats."""

    def __init__(self, favour=None, yes_no=(3.0, 0.0), box=(0.1, 0.2, 0.5, 0.6), stats=None, mirror_box=None):
        self.favour = favour or {}
        self.yes_no = yes_no
        self.box = list(box) if box else None
        self.mirror_box = mirror_box
        self.stats = stats or {"mean_p": 0.7, "min_p": 0.2, "mean_entropy": 0.6}
        self.calls = {"logits": 0, "ground": 0}
        self.prompts = []

    def next_token_logits(self, image, text, tokens):
        self.calls["logits"] += 1
        self.prompts.append(text)
        if list(tokens) == ["Yes", "No"]:
            return np.array(self.yes_no, float)
        target = self.favour.get("token")
        if "content" in self.favour:                                   # favour the letter of an option text
            for line in text.splitlines():
                if line.endswith(self.favour["content"]) and ". " in line:
                    target = line.split(".")[0]
        return np.array([4.0 if t == target else 0.0 for t in tokens])

    def ground(self, image, target):
        self.calls["ground"] += 1
        if self.calls["ground"] == 2 and self.mirror_box is not None:
            return list(self.mirror_box), self.stats
        return (list(self.box) if self.box else None), self.stats


def img():
    return Image.new("RGB", (64, 48), "white")


def calibration(threshold=0.6):
    cal = Calibration()
    cal.existence_platt = Logistic([0.0], [1.0], [1.0, 0.0])          # P(exists) = sigmoid(z)
    cal.box_first = Logistic([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0])   # always 0.5
    cal.box_full = Logistic([0.0] * 4, [1.0] * 4, [0.0, 0.0, 0.0, 4.0, -2.0])          # driven by agreement
    cal.fills = {"noul_z": 0.0, "box_conf": 0.7, "min_p": 0.2, "mean_entropy": 0.6, "agree": 0.7}
    cal.abstain_threshold = threshold
    return cal


# ---------------------------------------------------------------- schema

def test_schema_rejects_bad_requests():
    with pytest.raises(RequestError):
        parse_request({"questions": {}})
    with pytest.raises(RequestError):
        parse_request({"questions": {"q": {"type": "choice", "question": "x", "options": ["a"]}}})
    with pytest.raises(RequestError):
        parse_request({"questions": {"q": {"type": "box", "target": "dog"}}})              # box needs an image
    with pytest.raises(RequestError):
        parse_request({"questions": {"q": {"type": "noul", "question": "x", "image": "missing"}}})
    with pytest.raises(RequestError):
        parse_request({"questions": {"q": {"type": "essay", "question": "x"}}})


# ---------------------------------------------------------------- choice / score / noul

def test_choice_is_bounded_and_sums_to_one():
    d = Decider(FakeBackend(favour={"content": "pug"}), calibration())
    out = d.decide({"assets": {"i": img()},
                    "questions": {"q": {"type": "choice", "question": "Breed?", "options": ["beagle", "pug", "boxer"],
                                        "image": "i"}}})["answers"]["q"]
    assert out["choice"] == "pug"
    assert set(out["probabilities"]) == {"beagle", "pug", "boxer"}
    assert abs(sum(out["probabilities"].values()) - 1) < 1e-3


def test_rotation_follows_content_not_position():
    be = FakeBackend(favour={"content": "boxer"})
    d = Decider(be, calibration())
    out = d.choice(parse_request({"questions": {"q": {"type": "choice", "question": "Breed?",
                                                      "options": ["beagle", "pug", "boxer"], "rotate": True}}}
                                 ).questions["q"], None)
    assert out["choice"] == "boxer" and be.calls["logits"] == 3


def test_score_returns_level_and_expected_index():
    d = Decider(FakeBackend(favour={"token": "3"}), calibration())
    out = d.decide({"questions": {"q": {"type": "score", "question": "How fresh?",
                                        "levels": ["stale", "ok", "fresh"]}}})["answers"]["q"]
    assert out["level"] == "fresh" and out["index"] == 2 and out["expected_index"] > 1.5


def test_noul_probability_and_state_in_prompt():
    be = FakeBackend(yes_no=(2.0, 0.0))
    d = Decider(be, calibration())
    out = d.decide({"state": {"aisle": 4}, "questions": {"q": {"type": "noul", "question": "Is the shelf empty?"}}})
    a = out["answers"]["q"]
    assert a["answer"] is True and 0.5 < a["probability"] < 1
    assert '"aisle": 4' in be.prompts[0]


# ---------------------------------------------------------------- box cascade

def test_box_abstains_without_grounding_when_absent():
    be = FakeBackend(yes_no=(-3.0, 0.0))                                   # P(exists) ~ 0.05
    d = Decider(be, calibration())
    a = d.decide({"assets": {"i": img()}, "questions": {"q": {"type": "box", "target": "a cat", "image": "i"}}}
                 )["answers"]["q"]
    assert a["abstained"] and a["box"] is None and a["passes"] == 0 and be.calls["ground"] == 0


@pytest.mark.parametrize("preset,passes", [("fast", 1), ("balanced", 1), ("accurate", 2)])
def test_box_presets_control_the_mirror_pass(preset, passes):
    be = FakeBackend()
    cal = calibration()
    if preset == "balanced":                                               # first-pass 0.5 is below the band
        cal.box_first = Logistic([0.0] * 3, [1.0] * 3, [0.0, 0.0, 0.0, -3.0])
    a = Decider(be, cal, preset).decide({"assets": {"i": img()},
                                         "questions": {"q": {"type": "box", "target": "the mug", "image": "i"}}}
                                        )["answers"]["q"]
    assert not a["abstained"] and a["passes"] == passes and be.calls["ground"] == passes
    assert set(a["box"]) == {"left", "top", "right", "bottom"}


def test_mirror_agreement_raises_confidence():
    box = (0.1, 0.2, 0.5, 0.6)
    agree_box = (1 - 0.5, 0.2, 1 - 0.1, 0.6)                               # same box seen in the mirrored image
    lo = Decider(FakeBackend(box=box, mirror_box=(0.0, 0.0, 0.05, 0.05)), calibration(), "accurate")
    hi = Decider(FakeBackend(box=box, mirror_box=agree_box), calibration(), "accurate")
    req = {"assets": {"i": img()}, "questions": {"q": {"type": "box", "target": "the mug", "image": "i"}}}
    assert hi.decide(req)["answers"]["q"]["box_confidence"] > lo.decide(req)["answers"]["q"]["box_confidence"]


def test_unparsable_box_abstains():
    a = Decider(FakeBackend(box=None), calibration()).decide(
        {"assets": {"i": img()}, "questions": {"q": {"type": "box", "target": "x", "image": "i"}}})["answers"]["q"]
    assert a["abstained"] and a["box"] is None


# ---------------------------------------------------------------- helpers and calibration

def test_swap_left_right_and_iou():
    assert swap_left_right("cup on the Left of the right plate") == "cup on the right of the left plate"
    assert swap_left_right("leftover rice") == "leftover rice"
    assert box_iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0 and box_iou([0, 0, .5, .5], [.5, .5, 1, 1]) == 0.0


def test_fit_temperature_recovers_known_scale():
    rng = np.random.default_rng(0)
    rows, correct = [], []
    for _ in range(3000):
        z = rng.normal(0, 2, 4)
        correct.append(int(rng.choice(4, p=softmax(z))))
        rows.append(2.0 * z)                                                # over-sharp by 2x
    assert 1.6 < fit_temperature(rows, correct) < 2.5


def test_shipped_calibration_loads():
    cal = Calibration.default()
    assert cal.existence_platt is not None and cal.box_full is not None
    p = cal.box_full([[0.7, 0.2, 0.6, 0.9]])[0]
    assert 0 < p < 1


def test_fit_logistic_is_monotone_in_signal():
    rng = np.random.default_rng(1)
    x = rng.random(2000)
    y = rng.random(2000) < x
    m = fit_logistic(x[:, None], y)
    assert m([[0.9]])[0] > m([[0.1]])[0]
