"""Jev-style Box answer, step 3b: better box confidence, Platt-calibrated existence.

Changes from step 3 (same data and seed, so numbers compare directly):
  * box-confidence features: min / mean token probability and mean entropy over the
    generated coordinate tokens, plus AGREEMENT = IoU between the box and the box from a
    second pass on the horizontally mirrored image ("left" <-> "right" swapped in the
    expression, box mapped back); a logistic model fitted on the calibration split maps
    them to P(IoU>=0.5); single-feature AUROCs reported
  * existence: Platt scaling (a * z + b) instead of temperature only
  * abstention threshold also tuned on the calibration split (reported next to 0.5)
  * every item's raw readouts written to records.jsonl for offline recalibration

Step 3 notes:

Step 2 showed a head on frozen last-layer states cannot find boxes (Acc@0.5 0.17), while
the VLM's own generated boxes reach 0.86 with confidence already near-calibrated. So the
box should be READ from generation, made cheap, calibrated, and allowed to abstain:

Noul    "Is there {expr} in the image?"  P(yes) from the Yes / No token logits (one forward),
        temperature fitted on a calibration split
Box     prompt prefilled with the model's own answer format (```json [{"bbox_2d": [ ) so it
        only writes the four coordinates (~15 tokens); box = those numbers (resized-image
        pixels -> normalised); raw confidence = mean probability of the generated tokens up
        to the closing bracket; calibrated to P(IoU>=0.5) with a histogram map fitted on the
        calibration split
Decide  abstain if calibrated P(exists) < 0.5; else return the box with
        confidence = P(exists) * P(IoU>=0.5 | box confidence)

Data    lmms-lab/RefCOCO. Calibration: 1000 `val` items + 25% absent-target negatives.
        Evaluation: VB_N_EVAL items from testA + testB + 25% negatives. A negative pairs an
        image with an expression from another image.
Metrics present : Acc@IoU0.5, mean IoU, parse rate, box-confidence ECE (raw / calibrated),
                  accuracy at 50% / 80% coverage
        absent  : abstention rate; present: wrongful-abstention rate; existence AUROC
        end-to-end: correct = (absent and abstained) or (present, answered, IoU>=0.5);
                  accuracy and ECE of the final decision confidence
        speed   : seconds per item (Noul + box)

Outputs in /kaggle/working: vjev_box_readout2.json, records.jsonl
"""
import json
import math
import os
import random
import re
import subprocess
import sys
import time

import numpy as np
import torch

if os.path.exists("/kaggle"):                       # only install on Kaggle, never into a local env
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.49", "accelerate", "datasets"],
                   check=True)
from datasets import load_dataset  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402

MODEL = os.environ.get("VB_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
OUT = os.environ.get("VB_OUT", "/kaggle/working")
N_CAL = int(os.environ.get("VB_N_CAL", 1000))
N_EVAL = int(os.environ.get("VB_N_EVAL", 1600))
NEG_RATE = 0.25
PIX = 448 * 448
PREFILL = '```json\n[\n\t{"bbox_2d": ['
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()
random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------- data

def norm_item(ex):
    img = ex.get("image")
    if img is None:
        return None
    img = img.convert("RGB")
    w, h = img.size
    expr = ex.get("answer")
    if isinstance(expr, list):
        expr = expr[0] if expr else None
    if not expr:
        expr = ex.get("question")
    bb = ex.get("bbox")
    if not expr or bb is None or len(bb) != 4:
        return None
    x, y, bw, bh = [float(v) for v in bb]
    box = [x / w, y / h, (x + bw) / w, (y + bh) / h]
    if not (0 <= box[0] < box[2] <= 1.001 and 0 <= box[1] < box[3] <= 1.001):
        return None
    return img, str(expr).strip(), [min(max(v, 0.0), 1.0) for v in box]


def build(src, rng):
    """[(image, expression, box, exists)] with NEG_RATE absent-target negatives."""
    items = []
    for ds, i in src:
        got = norm_item(ds[i])
        if got:
            items.append((got[0], got[1], got[2], True))
    n_neg = int(len(items) * NEG_RATE)
    for _ in range(n_neg):
        a, b = rng.sample(range(len(items)), 2)
        items.append((items[a][0], items[b][1], items[a][2], False))
    rng.shuffle(items)
    return items


def load():
    rng = random.Random(SEED)
    val = load_dataset("lmms-lab/RefCOCO", split="val")
    vi = list(range(len(val)))
    rng.shuffle(vi)
    cal_src = [(val, i) for i in vi[:N_CAL]]
    test_src = []
    for split in ("testA", "testB"):
        ds = load_dataset("lmms-lab/RefCOCO", split=split)
        test_src += [(ds, i) for i in range(len(ds))]
    rng.shuffle(test_src)
    return build(cal_src, rng), build(test_src[:N_EVAL], rng)


# ---------------------------------------------------------------- model readouts

class VLM:
    def __init__(self):
        self.proc = AutoProcessor.from_pretrained(MODEL, min_pixels=PIX, max_pixels=PIX)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
        tok = self.proc.tokenizer
        self.yes_no = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No")]

    def _inputs(self, image, content_text, prefill=""):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": content_text}]}]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True) + prefill
        return self.proc(text=[prompt], images=[image], return_tensors="pt").to(DEV)

    @torch.no_grad()
    def noul(self, image, expr):
        inp = self._inputs(image, f"Is there {expr} in the image? Answer Yes or No.")
        z = self.model(**inp).logits[0, -1].float()[self.yes_no].cpu().numpy()
        return float(z[0] - z[1])

    @torch.no_grad()
    def box(self, image, expr):
        inp = self._inputs(image, f"Locate \"{expr}\" in the image and output its bbox_2d coordinates in JSON.", PREFILL)
        g = self.model.generate(**inp, max_new_tokens=24, do_sample=False, output_scores=True,
                                return_dict_in_generate=True)
        new = g.sequences[0, inp["input_ids"].shape[1]:]
        tok = self.proc.tokenizer
        text, logps, ents = "", [], []
        for s, t in zip(g.scores, new):
            lp = torch.log_softmax(s[0].float(), -1)
            logps.append(lp[t].item())
            ents.append(float(-(lp.exp() * lp).sum()))
            text += tok.decode([int(t)], skip_special_tokens=True)
            if "]" in text:
                break
        nums = re.findall(r"-?\d+\.?\d*", text.split("]")[0])
        conf = float(np.exp(np.mean(logps))) if logps else 0.0
        stats = {"mean_p": conf, "min_p": float(np.exp(min(logps))) if logps else 0.0,
                 "mean_entropy": float(np.mean(ents)) if ents else 0.0, "n_tokens": len(logps)}
        if len(nums) < 4:
            return None, stats
        _, gh, gw = inp["image_grid_thw"][0].tolist()
        W, H = gw * 14, gh * 14
        x0, y0, x1, y1 = [float(v) for v in nums[:4]]
        b = [x0 / W, y0 / H, x1 / W, y1 / H]
        return [min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])], stats


# ---------------------------------------------------------------- metrics

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return round(float(e), 4)


def fit_hist(conf, correct, bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    qs = np.quantile(conf, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    vals = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (conf > lo) & (conf <= hi)
        vals.append(correct[m].mean() if m.any() else (vals[-1] if vals else 0.0))
    vals = np.maximum.accumulate(np.array(vals))
    return lambda c: vals[np.clip(np.searchsorted(qs, np.asarray(c, float), side="left") - 1, 0, bins - 1)]


def fit_temperature(z, y):
    """Temperature for a yes/no logit difference, minimising NLL."""
    z, y = np.asarray(z, float), np.asarray(y, float)
    nll = lambda t: -np.mean(y * np.log(1 / (1 + np.exp(-z / t)) + 1e-12) + (1 - y) * np.log(1 - 1 / (1 + np.exp(-z / t)) + 1e-12))
    return float(min(np.exp(np.linspace(np.log(0.1), np.log(10), 120)), key=nll))


def fit_logistic(X, y, l2=1e-2, iters=3000, lr=0.1):
    """Plain gradient-descent logistic regression on standardised features."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0])
    return lambda Xn: 1 / (1 + np.exp(-(np.c_[(np.asarray(Xn, float) - mu) / sd, np.ones(len(Xn))] @ w)))


def coverage_acc(conf, correct, frac):
    order = np.argsort(-np.asarray(conf))
    keep = order[:max(1, int(len(order) * frac))]
    return round(float(np.asarray(correct, float)[keep].mean()), 4)


def auroc(score, label):
    score, label = np.asarray(score), np.asarray(label)
    pos, neg = score[label == 1], score[label == 0]
    if not len(pos) or not len(neg):
        return None
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return round(float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)


# ---------------------------------------------------------------- main

def swap_lr(expr):
    """Mirroring the image swaps which object 'left' / 'right' refers to."""
    return re.sub(r"\b(left|right)\b", lambda m: {"left": "right", "right": "left"}[m.group(1).lower()], expr,
                  flags=re.IGNORECASE)


def run(vlm, items, name):
    from PIL import ImageOps
    recs, t = [], time.time()
    for n, (img, expr, gt, exists) in enumerate(items):
        z = vlm.noul(img, expr)
        b, st = vlm.box(img, expr)
        bm, _ = vlm.box(ImageOps.mirror(img), swap_lr(expr))
        agree = 0.0
        if b is not None and bm is not None:
            agree = iou(b, [1 - bm[2], bm[1], 1 - bm[0], bm[3]])
        recs.append({"exists": exists, "noul_z": z, "box_conf": st["mean_p"], "min_p": st["min_p"],
                     "mean_entropy": st["mean_entropy"], "agree": agree, "parsed": b is not None,
                     "gen_tokens": st["n_tokens"], "box": b, "gt": gt, "expr": expr,
                     "iou": iou(b, gt) if (b is not None and exists) else 0.0})
        if n % 250 == 0:
            print(f"  {name} {n}/{len(items)}  {(time.time() - t) / max(n, 1):.2f}s/item  "
                  f"{(time.time() - T0) / 60:.0f} min", flush=True)
    return recs, (time.time() - t) / max(len(items), 1)


def main():
    cal_items, test_items = load()
    print(f"cal {len(cal_items)}  test {len(test_items)}  on {DEV}", flush=True)
    vlm = VLM()
    cal, _ = run(vlm, cal_items, "cal")
    test, secs = run(vlm, test_items, "test")

    # a handful of items can produce non-finite readouts; impute the calibration-split median
    # so they cannot poison the calibrators (found in the first run, see tools/recalibrate_box.py)
    for key in ("noul_z", "box_conf", "min_p", "mean_entropy", "agree"):
        vals = np.array([r[key] if r[key] is not None else np.nan for r in cal], float)
        fill = float(np.nanmedian(vals))
        for r in cal + test:
            if r[key] is None or not np.isfinite(r[key]):
                r[key] = fill
    with open(f"{OUT}/records.jsonl", "w") as f:
        for split, rs in (("cal", cal), ("test", test)):
            for r in rs:
                f.write(json.dumps({"split": split, **r}) + chr(10))

    feats = lambda rs: [[r["box_conf"], r["min_p"], r["mean_entropy"], r["agree"]] for r in rs]
    # existence: Platt scaling on the Noul logit
    platt = fit_logistic([[r["noul_z"]] for r in cal], [r["exists"] for r in cal])
    pe_cal = platt([[r["noul_z"]] for r in cal])
    pe_test = platt([[r["noul_z"]] for r in test])
    # box confidence: logistic on the four features, present targets only
    cal_pos = [r for r in cal if r["exists"]]
    box_model = fit_logistic(feats(cal_pos), [r["iou"] >= 0.5 for r in cal_pos])

    pos = [r for r in test if r["exists"]]
    neg = [r for r in test if not r["exists"]]
    correct = np.array([r["iou"] >= 0.5 for r in pos], float)
    pb_pos = box_model(feats(pos))

    def end_to_end(threshold, recs, pe):
        pb = box_model(feats(recs))
        ok, conf = [], []
        for r, e, b in zip(recs, pe, pb):
            if e < threshold:
                ok.append(not r["exists"])
                conf.append(1 - e)
            else:
                ok.append(r["exists"] and r["iou"] >= 0.5)
                conf.append(e * b)
        return np.array(ok, float), np.array(conf)

    tuned = max(np.linspace(0.05, 0.95, 37), key=lambda th: end_to_end(th, cal, pe_cal)[0].mean())
    single = {k: auroc([-r[k] if k == "mean_entropy" else r[k] for r in pos], correct.astype(int))
              for k in ("box_conf", "min_p", "mean_entropy", "agree")}
    res = {
        "present": {"n": len(pos), "acc_iou0.5": round(float(correct.mean()), 4),
                    "mean_iou": round(float(np.mean([r["iou"] for r in pos])), 4),
                    "parse_rate": round(float(np.mean([r["parsed"] for r in pos])), 4),
                    "box_conf_ece_raw_mean_p": ece([r["box_conf"] for r in pos], correct),
                    "box_conf_ece_calibrated": ece(pb_pos, correct),
                    "box_conf_auroc": auroc(pb_pos, correct.astype(int)),
                    "single_feature_auroc": single,
                    "acc_at_coverage_0.8": coverage_acc(pb_pos, correct, 0.8),
                    "acc_at_coverage_0.5": coverage_acc(pb_pos, correct, 0.5),
                    "mean_generated_tokens": round(float(np.mean([r["gen_tokens"] for r in pos])), 1)},
        "existence": {"auroc": auroc(pe_test, [int(r["exists"]) for r in test]),
                      "ece": ece(pe_test, [r["exists"] for r in test])},
        "seconds_per_item": round(secs, 3),
    }
    for label, th in (("threshold_0.5", 0.5), ("threshold_tuned", float(tuned))):
        ok, conf = end_to_end(th, test, pe_test)
        res[f"end_to_end_{label}"] = {
            "threshold": round(th, 3), "accuracy": round(float(ok.mean()), 4), "decision_conf_ece": ece(conf, ok),
            "acc_at_coverage_0.8": coverage_acc(conf, ok, 0.8),
            "abstain_rate_absent": round(float(np.mean([e < th for e, r in zip(pe_test, test) if not r["exists"]])), 4) if neg else None,
            "wrongful_abstain_rate_present": round(float(np.mean([e < th for e, r in zip(pe_test, test) if r["exists"]])), 4)}
    res["box_only_accuracy_if_never_abstaining"] = round(float(np.mean([r["exists"] and r["iou"] >= 0.5 for r in test])), 4)
    out = {"model": MODEL, "data": "lmms-lab/RefCOCO (cal: val; eval: testA+testB)", "neg_rate": NEG_RATE,
           "n_cal": len(cal), "n_test": len(test), "prefill": PREFILL, "results": res,
           "total_minutes": round((time.time() - T0) / 60, 1)}
    json.dump(out, open(f"{OUT}/vjev_box_readout2.json", "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
