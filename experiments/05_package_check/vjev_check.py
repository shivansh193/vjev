"""End-to-end check of the packaged vjev API on the real model (GPU).

Loads the package from the private `vjev-src` dataset, then calls Decider.decide() exactly
as a user would:
  * box  : RefCOCO testA, 300 present targets + 75 absent (expression from another image),
           presets fast and balanced
  * choice: Oxford-IIIT Pets, 100 images, 5-way
  * one multi-question request (choice + score + noul + box on one image) for latency
Expected (from the experiments): box Acc@0.5 ~0.86 on present targets, abstention on most
absent targets, choice ~0.97.

Outputs in /kaggle/working: vjev_check.json
"""
import glob
import json
import os
import random
import subprocess
import sys
import time
import zipfile

import numpy as np

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.49", "accelerate", "datasets"],
               check=True)

# ---- load the package from the dataset (zip or folder)
pkg = glob.glob("/kaggle/input/**/decider.py", recursive=True)
if not pkg:
    z = glob.glob("/kaggle/input/**/vjev.zip", recursive=True)
    zipfile.ZipFile(z[0]).extractall("/kaggle/tmp/vjevpkg")
    pkg = glob.glob("/kaggle/tmp/vjevpkg/**/decider.py", recursive=True)
root = os.path.dirname(os.path.dirname(pkg[0]))
sys.path.insert(0, root)
import vjev  # noqa: E402
from vjev import Decider  # noqa: E402
from vjev.backend import QwenVLBackend  # noqa: E402
from vjev.decider import box_iou  # noqa: E402

print("vjev", vjev.__version__, "from", root, flush=True)
OUT = "/kaggle/working"
rng = random.Random(1)


def refcoco_items(n_present=300, neg_rate=0.25):
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/RefCOCO", split="testA")
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    items = []
    for i in idx:
        ex = ds[i]
        img = ex["image"].convert("RGB")
        w, h = img.size
        expr = ex["answer"][0] if isinstance(ex["answer"], list) else ex["answer"]
        x, y, bw, bh = [float(v) for v in ex["bbox"]]
        items.append((img, str(expr), [x / w, y / h, (x + bw) / w, (y + bh) / h], True))
        if len(items) == n_present:
            break
    for _ in range(int(n_present * neg_rate)):
        a, b = rng.sample(range(n_present), 2)
        items.append((items[a][0], items[b][1], items[a][2], False))
    return items


def run_box(decider, items):
    rows, t0 = [], time.time()
    for img, expr, gt, exists in items:
        a = decider.decide({"assets": {"i": img}, "questions": {"q": {"type": "box", "target": expr, "image": "i"}}}
                           )["answers"]["q"]
        b = a["box"]
        iou = box_iou([b["left"], b["top"], b["right"], b["bottom"]], gt) if (b and exists) else 0.0
        rows.append({"exists": exists, "abstained": a["abstained"], "iou": iou, "conf": a["confidence"],
                     "passes": a["passes"]})
    pos = [r for r in rows if r["exists"]]
    neg = [r for r in rows if not r["exists"]]
    answered = [r for r in pos if not r["abstained"]]
    ok = [(not r["exists"] and r["abstained"]) or (r["exists"] and not r["abstained"] and r["iou"] >= 0.5) for r in rows]
    return {"acc_iou0.5_present_answered": round(float(np.mean([r["iou"] >= 0.5 for r in answered])), 4),
            "acc_iou0.5_present_all": round(float(np.mean([r["iou"] >= 0.5 for r in pos])), 4),
            "abstain_rate_absent": round(float(np.mean([r["abstained"] for r in neg])), 4),
            "wrongful_abstain_rate_present": round(float(np.mean([r["abstained"] for r in pos])), 4),
            "end_to_end_accuracy": round(float(np.mean(ok)), 4),
            "mean_passes": round(float(np.mean([r["passes"] for r in rows])), 3),
            "seconds_per_item": round((time.time() - t0) / len(rows), 3), "n": len(rows)}


def run_choice(decider, n=100):
    import torchvision
    ds = torchvision.datasets.OxfordIIITPet("/kaggle/tmp/pets", split="test", download=True)
    names = [c.replace("_", " ") for c in ds.classes]
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    hits, t0 = [], time.time()
    for i in idx[:n]:
        img, label = ds[i]
        opts = [label] + rng.sample([c for c in range(len(names)) if c != label], 4)
        rng.shuffle(opts)
        a = decider.decide({"assets": {"i": img}, "questions": {"q": {
            "type": "choice", "question": "Which breed of pet is shown in the image?",
            "options": [names[o] for o in opts], "image": "i"}}})["answers"]["q"]
        hits.append(a["choice"] == names[label])
    return {"accuracy": round(float(np.mean(hits)), 4), "seconds_per_item": round((time.time() - t0) / n, 3), "n": n}


def main():
    t0 = time.time()
    backend = QwenVLBackend()
    items = refcoco_items()
    res = {"box": {}}
    for preset in ("fast", "balanced"):
        res["box"][preset] = run_box(Decider(backend, preset=preset), items)
        print(preset, json.dumps(res["box"][preset]), flush=True)
    res["choice_pets"] = run_choice(Decider(backend))
    print("choice", json.dumps(res["choice_pets"]), flush=True)

    img = items[0][0]
    ts = time.time()
    multi = Decider(backend).decide({"state": {"source": "RefCOCO"}, "assets": {"i": img}, "questions": {
        "kind": {"type": "choice", "question": "What is the main subject?", "options": ["person", "animal", "vehicle", "food", "furniture"], "image": "i"},
        "crowd": {"type": "score", "question": "How crowded is the scene?", "levels": ["empty", "few", "some", "many"], "image": "i"},
        "outdoor": {"type": "noul", "question": "Is this photo taken outdoors?", "image": "i"},
        "target": {"type": "box", "target": items[0][1], "image": "i"}}})
    res["multi_question_request"] = {"answers": multi["answers"], "seconds": round(time.time() - ts, 3)}
    res["total_minutes"] = round((time.time() - t0) / 60, 1)
    json.dump(res, open(f"{OUT}/vjev_check.json", "w"), indent=2)
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
