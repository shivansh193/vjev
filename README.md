# vjev

Open, Jev-style **typed decisions about images**, training-free, on a small open
vision-language model (Qwen2.5-VL-3B by default).

You send images, optional JSON state and named, typed questions. You get back answers that
cannot leave their schema, with probabilities that are calibrated:

| Type | You give | You get |
|---|---|---|
| `choice` | question + 2–26 options | one option, probability per option |
| `score` | question + 2–9 ordered levels | a level, its index, expected index, distribution |
| `noul` | a yes/no question | `answer` and `P(yes)` |
| `box` | a referring expression | a box (left/top/right/bottom, fractions of the image) **or an explicit abstention**, with calibrated confidence |

Nothing is parsed from free text except the four box coordinates. Choice, score and noul
answers are softmaxes over the option tokens at the answer position. A box is the model's
own short grounding output (the answer format is prefilled, so it writes only the
coordinates), with confidence from token-level uncertainty and an optional mirrored-image
agreement check, gated by an existence question so it can say "not there".

Not affiliated with TypeSafe AI or Jev.

## Install and use

```bash
pip install -e ".[model]"            # add [server] for the HTTP server, [dev] for tests
```

```python
from vjev import Decider

d = Decider(preset="balanced")       # loads Qwen/Qwen2.5-VL-3B-Instruct
out = d.decide({
    "state": {"store": "aisle 4"},
    "assets": {"shelf": "shelf.jpg"},                 # PIL image, path, bytes or base64
    "questions": {
        "brand": {"type": "choice", "question": "Which brand is on the top shelf?",
                  "options": ["Lays", "Kurkure", "Bingo"], "image": "shelf"},
        "stock": {"type": "score", "question": "How well stocked is the shelf?",
                  "levels": ["empty", "sparse", "half", "full"], "image": "shelf"},
        "gap":   {"type": "noul", "question": "Is there an empty slot on the shelf?", "image": "shelf"},
        "mug":   {"type": "box", "target": "the red packet on the left", "image": "shelf"},
    },
})
out["answers"]["mug"]
# {"type": "box", "abstained": False, "box": {"left": .., "top": .., "right": .., "bottom": ..},
#  "exists": 0.97, "confidence": 0.91, "box_confidence": 0.94, "passes": 1}
```

HTTP server: `python -m vjev.server --preset balanced`, then `POST /v1/decide` with the same
JSON (images as base64 strings).

### Presets (box only)

| Preset | Mirrored second pass | Est. s/item on a T4* | Acc. at 80% coverage | Confidence AUROC |
|---|---|---|---|---|
| `fast` | never | 1.5 | 0.892 | 0.787 |
| `balanced` | when first-pass confidence is 0.6–0.9 (~50%) | 2.1 | 0.908 | 0.811 |
| `accurate` | always | 2.7 | 0.910 | 0.828 |

\*Estimated from measured per-pass timings, unbatched; RefCOCO testA+testB, 1600 present
and 400 absent targets. Absent targets cost one forward pass (the existence check) and skip
grounding.

## Results so far (Qwen2.5-VL-3B-Instruct, frozen, no training)

**Choice / noul** (Oxford-IIIT Pets, 1000 images, 5-way choice):

| | Accuracy | ECE raw | ECE calibrated |
|---|---|---|---|
| choice | 96.8% | 0.021 | **0.004** |
| noul | 96.8% | 0.056 | **0.012** |

**Box** (RefCOCO testA + testB, 2000 items, 20% absent targets):

| | Native generation | **vjev box** |
|---|---|---|
| Acc@IoU 0.5 (present) | 85.6% | **86.3%** |
| Confidence ECE | 0.022 | **0.026** |
| Accuracy of the 50% most confident | 93.6% | **96.0%** |
| Can abstain on absent targets | no | **yes: 78% of absent, 3% wrongly on present** |
| End-to-end accuracy (correct box or correct abstention) | 69.1% | **83.0%** |
| Existence AUROC / ECE | n/a | 0.946 / 0.027 |

What did *not* work: a trained head (pointer over image tokens + edge distributions) on
frozen last-layer states reached only 16.8% Acc@0.5. The location is produced during
generation, not linearly present in those states, which is why the box is read from the
model's own grounding.

**Packaged API on the real model** (`experiments/05_package_check`, RefCOCO testA, 300 present
+ 75 absent): box Acc@0.5 0.88 (0.917 when it answers), wrongful abstention 4%,
measured 1.57 s/item (`fast`) and 2.07 s/item (`balanced`); Pets choice 99% at 0.34 s.
Absent-target abstention there was 49%: testA expressions all describe people, so a
borrowed expression is often actually present in the image. A cleaner absent-target
benchmark is in progress.

## How it was built

Each experiment is a Kaggle script kernel (`kaggle kernels push -p experiments/<dir>`) with
its downloaded results in `output/`:

| Dir | Question | Outcome |
|---|---|---|
| `01_choice_noul_readout` | Can option-token readout give calibrated typed answers on images? | Yes: 96.8% / ECE 0.004 (choice), 0.012 (noul) after one temperature |
| `02_box_head_negative_result` | Can a small head on frozen last-layer states predict boxes? | No: 16.8% Acc@0.5; location is produced during generation |
| `03_box_readout` | Read the model's own grounding (prefilled format) + existence gate | Native-level accuracy, abstention works; token-probability confidence saturates |
| `04_box_readout_calibrated` | Better confidence: token uncertainty + mirrored-image agreement, Platt existence | Conf ECE 0.026, 96% at 50% coverage, end-to-end 83% |
| `05_package_check` | Does the packaged API reproduce it on the real model? | Yes (see above) |

`tools/recalibrate_box.py` refits the calibrators offline from saved per-item records;
`tools/cascade_box.py` simulates the mirror-pass cascade and writes the shipped calibration file.

## Calibration

Shipped calibrators (`vjev/calibration/`) were fitted on held-out splits (Pets for
choice/noul temperatures; RefCOCO val for existence and box confidence). They will drift on
other domains. Refit on a few hundred labelled examples with `fit_temperature` /
`fit_logistic` and pass your own `Calibration` to `Decider`.

## Limitations

* English only; single image per question; still images.
* Box confidence comes from one model's token statistics; very different image domains
  (e.g. medical, top-down CCTV) need recalibration.
* RefCOCO "absent" targets were made by borrowing another image's expression, so some are
  accidentally present; abstention numbers are slightly pessimistic.
* Not a safety, medical, legal or identity tool.

## Licence

Apache-2.0 for this code. The default model (Qwen2.5-VL-3B-Instruct) has its own licence;
check it before commercial use. Benchmark datasets keep their own licences.
