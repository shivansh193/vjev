"""Simulate the cascaded Box answer offline from saved per-item records (no GPU).

Order of work per question:
  1. Noul  : P(exists) (Platt). If below the abstain threshold -> answer "absent", stop.
  2. Box   : one grounding pass -> first-pass confidence p1 from token features
             (mean / min token probability, mean entropy).
  3. Mirror: only if p1 falls inside the uncertain band [lo, hi]; then the full confidence
             p2 (first-pass features + mirror agreement) replaces p1.
The band is chosen on the calibration split: the cheapest band whose 80%-coverage accuracy
is within `--tolerance` of always running the mirror pass. Costs come from the measured
per-pass timings (seconds), so the reported time per item is an estimate.

Usage:
    python tools/cascade_box.py experiments/04_box_readout_calibrated/output/records.jsonl --out cascade.json
"""
import argparse
import json

import numpy as np

FIRST = ["box_conf", "min_p", "mean_entropy"]
FULL = FIRST + ["agree"]


def fit_logistic(X, y, l2=1e-2, iters=4000, lr=0.1):
    X, y = np.asarray(X, float), np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0])
    return {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist()}


def apply(model, X):
    X = np.asarray(X, float)
    Z = np.c_[(X - np.array(model["mu"])) / np.array(model["sd"]), np.ones(len(X))]
    return 1 / (1 + np.exp(-Z @ np.array(model["w"])))


def clean(recs, keys):
    """Median-impute non-finite readouts (fitted on the given records)."""
    fills = {}
    for k in keys:
        v = np.array([r.get(k) if r.get(k) is not None else np.nan for r in recs], float)
        fills[k] = float(np.nanmedian(v))
    return fills


def matrix(recs, keys, fills):
    X = np.array([[r.get(k) if r.get(k) is not None else np.nan for k in keys] for r in recs], float)
    for j, k in enumerate(keys):
        X[~np.isfinite(X[:, j]), j] = fills[k]
    return X


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = ((conf >= lo) if i == 0 else (conf > lo)) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return round(float(e), 4)


def coverage_acc(conf, correct, frac):
    order = np.argsort(-np.asarray(conf, float))
    keep = order[:max(1, int(len(order) * frac))]
    return round(float(np.asarray(correct, float)[keep].mean()), 4)


def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label)
    pos, neg = score[label == 1], score[label == 0]
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return round(float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)


class Cascade:
    def __init__(self, cal, abstain_threshold):
        self.fills = clean(cal, ["noul_z"] + FULL)
        self.platt = fit_logistic(matrix(cal, ["noul_z"], self.fills), [r["exists"] for r in cal])
        pos = [r for r in cal if r["exists"]]
        y = [r["iou"] >= 0.5 for r in pos]
        self.first = fit_logistic(matrix(pos, FIRST, self.fills), y)
        self.full = fit_logistic(matrix(pos, FULL, self.fills), y)
        self.threshold = abstain_threshold

    def decide(self, recs, band):
        """Per record: (correct, confidence, passes) where passes = box passes used."""
        lo, hi = band
        pe = apply(self.platt, matrix(recs, ["noul_z"], self.fills))
        p1 = apply(self.first, matrix(recs, FIRST, self.fills))
        p2 = apply(self.full, matrix(recs, FULL, self.fills))
        out = []
        for r, e, a, b in zip(recs, pe, p1, p2):
            if e < self.threshold:
                out.append((not r["exists"], 1 - e, 0))
                continue
            mirrored = lo <= a <= hi
            pb = b if mirrored else a
            out.append((r["exists"] and r["iou"] >= 0.5, e * pb, 2 if mirrored else 1))
        ok, conf, passes = (np.array(x, float) for x in zip(*out))
        return ok, conf, passes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--abstain", type=float, default=0.6, help="P(exists) below this -> abstain")
    ap.add_argument("--noul_s", type=float, default=0.30, help="seconds per Noul readout")
    ap.add_argument("--box_s", type=float, default=1.50, help="seconds per box pass")
    ap.add_argument("--tolerance", type=float, default=0.005)
    ap.add_argument("--out")
    ap.add_argument("--save_calibration", help="write the fitted calibrators (for the vjev package)")
    args = ap.parse_args()

    recs = [json.loads(line) for line in open(args.records)]
    cal = [r for r in recs if r["split"] == "cal"]
    test = [r for r in recs if r["split"] == "test"]
    cas = Cascade(cal, args.abstain)
    cost = lambda passes: args.noul_s + args.box_s * float(np.mean(passes))

    ref_ok, ref_conf, _ = cas.decide(cal, (0.0, 1.0))                    # always mirror
    ref = coverage_acc(ref_conf, ref_ok, 0.8)
    grid = [(lo, hi) for lo in np.linspace(0.0, 0.9, 19) for hi in np.linspace(0.1, 1.0, 19) if hi > lo]
    grid.append((1.1, 1.1))                                              # never mirror
    best = None
    for band in grid:
        ok, conf, passes = cas.decide(cal, band)
        if coverage_acc(conf, ok, 0.8) >= ref - args.tolerance:
            c = cost(passes)
            if best is None or c < best[0]:
                best = (c, band)
    band = best[1]

    def report(b):
        ok, conf, passes = cas.decide(test, b)
        pos = np.array([r["exists"] for r in test])
        return {"band": [round(x, 3) for x in b], "accuracy": round(float(ok.mean()), 4),
                "decision_conf_ece": ece(conf, ok), "decision_auroc": auroc(conf, ok.astype(int)),
                "acc_at_coverage_0.8": coverage_acc(conf, ok, 0.8),
                "acc_at_coverage_0.5": coverage_acc(conf, ok, 0.5),
                "mirror_rate_when_answering": round(float((passes == 2).sum() / max((passes > 0).sum(), 1)), 4),
                "abstain_rate": round(float((passes == 0).mean()), 4),
                "abstain_rate_absent": round(float((passes[~pos] == 0).mean()), 4),
                "wrongful_abstain_rate_present": round(float((passes[pos] == 0).mean()), 4),
                "est_seconds_per_item": round(cost(passes), 3)}

    res = {"abstain_threshold": args.abstain, "timing_assumed": {"noul_s": args.noul_s, "box_s": args.box_s},
           "cascade": report(band), "always_mirror": report((0.0, 1.0)), "never_mirror": report((1.1, 1.1))}
    print(json.dumps(res, indent=1))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)
    if args.save_calibration:
        json.dump({"source": args.records, "fills": cas.fills, "existence_platt": cas.platt,
                   "box_first_pass": cas.first, "box_full": cas.full, "first_features": FIRST,
                   "full_features": FULL, "abstain_threshold": args.abstain,
                   "mirror_band": [float(x) for x in band]}, open(args.save_calibration, "w"), indent=2)


if __name__ == "__main__":
    main()
