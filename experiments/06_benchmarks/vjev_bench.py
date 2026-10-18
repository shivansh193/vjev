"""Harder benchmarks for the packaged vjev API (GPU, ~1 h on a T4).

A. Clean absent targets (box abstention)
   RefCOCO images are COCO train2014 images. Using COCO instance annotations, an "absent"
   question borrows another image's expression ONLY if the COCO category that expression
   names is not annotated anywhere in this image (and the expression names a recognisable
   category). Present targets: RefCOCO testA + testB. Preset: fast. Shipped calibration.
B. Harder choice
   Oxford-IIIT Pets 10-way and 26-way (26 = one letter token per option), and A-OKVQA
   multiple choice (val split, 4 options). Temperature fitted on one half, metrics on the
   other half; raw numbers reported too.

Outputs in /kaggle/working: vjev_bench.json
"""
import glob
import json
import os
import random
import re
import subprocess
import sys
import time
import zipfile

import numpy as np

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.49", "accelerate", "datasets"],
               check=True)

pkg = glob.glob("/kaggle/input/**/decider.py", recursive=True)
if not pkg:
    z = glob.glob("/kaggle/input/**/vjev.zip", recursive=True)
    zipfile.ZipFile(z[0]).extractall("/kaggle/tmp/vjevpkg")
    pkg = glob.glob("/kaggle/tmp/vjevpkg/**/decider.py", recursive=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(pkg[0])))
from vjev import Decider, fit_temperature  # noqa: E402
from vjev.backend import QwenVLBackend  # noqa: E402
from vjev.calibration import softmax  # noqa: E402
from vjev.decider import LETTERS, box_iou  # noqa: E402

OUT = "/kaggle/working"
TMP = "/kaggle/tmp/bench"
rng = random.Random(7)
os.makedirs(TMP, exist_ok=True)

# COCO category names plus common referring-expression synonyms -> category
SYNONYMS = {
    "person": ["person", "man", "woman", "guy", "lady", "boy", "girl", "kid", "child", "player", "people", "baby"],
    "dog": ["dog", "puppy"], "cat": ["cat", "kitten"], "horse": ["horse"], "sheep": ["sheep", "lamb"],
    "cow": ["cow", "cattle"], "elephant": ["elephant"], "bear": ["bear"], "zebra": ["zebra"], "giraffe": ["giraffe"],
    "bird": ["bird", "duck", "pigeon"], "car": ["car", "taxi"], "bus": ["bus"], "truck": ["truck"],
    "motorcycle": ["motorcycle", "motorbike"], "bicycle": ["bicycle", "bike"], "airplane": ["airplane", "plane"],
    "train": ["train"], "boat": ["boat"], "chair": ["chair"], "couch": ["couch", "sofa"], "bed": ["bed"],
    "dining table": ["table"], "tv": ["tv", "television"], "laptop": ["laptop"], "cake": ["cake"],
    "pizza": ["pizza"], "donut": ["donut", "doughnut"], "sandwich": ["sandwich"], "banana": ["banana"],
    "apple": ["apple"], "orange": ["orange"], "broccoli": ["broccoli"], "carrot": ["carrot"], "bowl": ["bowl"],
    "cup": ["cup", "mug"], "bottle": ["bottle"], "vase": ["vase"], "umbrella": ["umbrella"],
    "suitcase": ["suitcase", "luggage"], "teddy bear": ["teddy"], "potted plant": ["plant"], "clock": ["clock"],
    "toilet": ["toilet"], "sink": ["sink"], "refrigerator": ["fridge", "refrigerator"], "oven": ["oven"],
}


def category_of(expr):
    words = set(re.findall(r"[a-z]+", expr.lower()))
    hits = [c for c, syn in SYNONYMS.items() if words & set(syn)]
    return hits[0] if len(hits) == 1 else None          # ambiguous or none -> unusable


def coco_categories_by_image():
    path = f"{TMP}/instances_train2014.json"
    if not os.path.exists(path):
        zp = f"{TMP}/ann.zip"
        subprocess.run(["wget", "-q", "-O", zp,
                        "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"], check=True)
        with zipfile.ZipFile(zp) as z:
            z.extract("annotations/instances_train2014.json", TMP)
        os.replace(f"{TMP}/annotations/instances_train2014.json", path)
    d = json.load(open(path))
    names = {c["id"]: c["name"] for c in d["categories"]}
    by_image = {}
    for a in d["annotations"]:
        by_image.setdefault(a["image_id"], set()).add(names[a["category_id"]])
    return by_image


def image_id(file_name):
    m = re.search(r"(\d{6,})", file_name or "")
    return int(m.group(1)) if m else None


def bench_absent(decider, n_present=800, n_absent=400):
    from datasets import load_dataset
    by_image = coco_categories_by_image()
    pool = []
    for split in ("testA", "testB"):
        ds = load_dataset("lmms-lab/RefCOCO", split=split)
        pool += [(ds, i) for i in range(len(ds))]
    rng.shuffle(pool)
    present, exprs = [], []
    for ds, i in pool:
        ex = ds[i]
        iid = image_id(ex.get("file_name"))
        if iid not in by_image:
            continue
        img = ex["image"].convert("RGB")
        w, h = img.size
        expr = ex["answer"][0] if isinstance(ex["answer"], list) else ex["answer"]
        x, y, bw, bh = [float(v) for v in ex["bbox"]]
        rec = (img, str(expr), [x / w, y / h, (x + bw) / w, (y + bh) / h], iid)
        exprs.append((str(expr), category_of(str(expr))))
        if len(present) < n_present:
            present.append(rec)
        if len(present) >= n_present and len(exprs) >= 4 * n_present:
            break
    absent, tries = [], 0
    while len(absent) < n_absent and tries < 50 * n_absent:
        tries += 1
        img, _, _, iid = rng.choice(present)
        expr, cat = rng.choice(exprs)
        if cat is not None and cat not in by_image[iid]:
            absent.append((img, expr, cat))
    rows, t0 = [], time.time()
    for img, expr, gt, _ in present:
        a = decider.decide({"assets": {"i": img}, "questions": {"q": {"type": "box", "target": expr, "image": "i"}}})["answers"]["q"]
        b = a["box"]
        rows.append({"exists": True, "abstained": a["abstained"], "exists_p": a["exists"],
                     "iou": box_iou([b["left"], b["top"], b["right"], b["bottom"]], gt) if b else 0.0})
    for img, expr, cat in absent:
        a = decider.decide({"assets": {"i": img}, "questions": {"q": {"type": "box", "target": expr, "image": "i"}}})["answers"]["q"]
        rows.append({"exists": False, "abstained": a["abstained"], "exists_p": a["exists"], "iou": 0.0})
    pos = [r for r in rows if r["exists"]]
    neg = [r for r in rows if not r["exists"]]
    ok = [(not r["exists"] and r["abstained"]) or (r["exists"] and not r["abstained"] and r["iou"] >= 0.5) for r in rows]
    s = np.array([r["exists_p"] for r in rows])
    lab = np.array([int(r["exists"]) for r in rows])
    ranks = np.argsort(np.argsort(s)) + 1
    auc = (ranks[lab == 1].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (len(lab) - lab.sum()))
    return {"n_present": len(pos), "n_absent_clean": len(neg),
            "acc_iou0.5_present_all": round(float(np.mean([r["iou"] >= 0.5 for r in pos])), 4),
            "abstain_rate_absent_clean": round(float(np.mean([r["abstained"] for r in neg])), 4),
            "wrongful_abstain_rate_present": round(float(np.mean([r["abstained"] for r in pos])), 4),
            "existence_auroc": round(float(auc), 4),
            "end_to_end_accuracy": round(float(np.mean(ok)), 4),
            "seconds_per_item": round((time.time() - t0) / len(rows), 3)}


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return round(float(e), 4)


def choice_logits(backend, image, question, options):
    letters = list(LETTERS[:len(options)])
    text = (f"{question}\nOptions:\n" + "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
            + "\nAnswer with the letter only.")
    return backend.next_token_logits(image, text, letters)


def evaluate_choice(items):
    """items: list of (logits, correct_idx). Fit T on the first half, report on the second."""
    half = len(items) // 2
    t = fit_temperature([z for z, _ in items[:half]], [c for _, c in items[:half]])
    out = {}
    for name, temp in (("raw", 1.0), ("calibrated", t)):
        probs = [softmax(z, temp) for z, _ in items[half:]]
        pred = [int(p.argmax()) for p in probs]
        correct = [p == c for p, (_, c) in zip(pred, items[half:])]
        out[name] = {"accuracy": round(float(np.mean(correct)), 4),
                     "ece": ece([p.max() for p in probs], correct)}
    out["temperature"] = round(t, 3)
    out["n_eval"] = len(items) - half
    return out


def bench_pets(backend, k, n=600):
    import torchvision
    ds = torchvision.datasets.OxfordIIITPet(f"{TMP}/pets", split="test", download=True)
    names = [c.replace("_", " ") for c in ds.classes]
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    items = []
    for i in idx[:n]:
        img, label = ds[i]
        opts = [label] + rng.sample([c for c in range(len(names)) if c != label], k - 1)
        rng.shuffle(opts)
        z = choice_logits(backend, img, "Which breed of pet is shown in the image?", [names[o] for o in opts])
        items.append((z, opts.index(label)))
    return evaluate_choice(items)


def bench_aokvqa(backend, n=1100):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceM4/A-OKVQA", split="validation")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    items = []
    for i in idx[:n]:
        ex = ds[i]
        z = choice_logits(backend, ex["image"].convert("RGB"), ex["question"], list(ex["choices"]))
        items.append((z, int(ex["correct_choice_idx"])))
    return evaluate_choice(items)


def main():
    t0 = time.time()
    backend = QwenVLBackend()
    res = {}
    res["choice_pets_10way"] = bench_pets(backend, 10)
    print("pets10", res["choice_pets_10way"], flush=True)
    res["choice_pets_26way"] = bench_pets(backend, 26)
    print("pets26", res["choice_pets_26way"], flush=True)
    try:
        res["choice_aokvqa_mc"] = bench_aokvqa(backend)
    except Exception as e:                               # dataset access can fail; keep the rest
        res["choice_aokvqa_mc"] = {"error": repr(e)[:300]}
    print("aokvqa", res["choice_aokvqa_mc"], flush=True)
    res["box_clean_absent"] = bench_absent(Decider(backend, preset="fast"))
    print("absent", res["box_clean_absent"], flush=True)
    res["total_minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(res, open(f"{OUT}/vjev_bench.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
