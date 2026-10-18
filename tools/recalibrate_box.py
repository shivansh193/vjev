"""Recompute the Jev-style Box metrics offline from saved per-item records (no GPU).

Fits the existence (Platt) and box-confidence (logistic) calibrators on the `cal` split
and evaluates on `test`, exactly like vjev_box_readout2.py, but with the few non-finite
readouts imputed by the calibration-split median instead of poisoning the fit.

Usage:
    python tools/recalibrate_box.py experiments/04_box_readout_calibrated/output/records.jsonl [--out results.json]
"""
import argparse
import json

import numpy as np

FEATURES = ["box_conf", "min_p", "mean_entropy", "agree"]


def ece(conf, correct, bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    assert np.isfinite(conf).all(), "non-finite confidence"
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return round(float(e), 4)


def auroc(score, label):
    score, label = np.asarray(score, float), np.asarray(label)
    pos, neg = score[label == 1], score[label == 0]
    if not len(pos) or not len(neg):
        return None
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return round(float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 4)


def coverage_acc(conf, correct, frac):
    order = np.argsort(-np.asarray(conf, float))
    keep = order[:max(1, int(len(order) * frac))]
    return round(float(np.asarray(correct, float)[keep].mean()), 4)


def fit_logistic(X, y, l2=1e-2, iters=4000, lr=0.1):
    X, y = np.asarray(X, float), np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * np.r_[w[:-1], 0])
    return lambda Xn: 1 / (1 + np.exp(-(np.c_[(np.asarray(Xn, float) - mu) / sd, np.ones(len(Xn))] @ w)))


def matrix(recs, keys, fill):
    X = np.array([[r.get(k) if r.get(k) is not None else np.nan for k in keys] for r in recs], float)
    bad = ~np.isfinite(X)
    X[bad] = np.take(fill, np.nonzero(bad)[1])
    return X, int(bad.any(1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--out")
    args = ap.parse_args()
    recs = [json.loads(line) for line in open(args.records)]
    cal = [r for r in recs if r["split"] == "cal"]
    test = [r for r in recs if r["split"] == "test"]

    fill_z = np.nanmedian([r["noul_z"] if r["noul_z"] is not None else np.nan for r in cal])
    Zc, nz_c = matrix(cal, ["noul_z"], np.array([fill_z]))
    Zt, nz_t = matrix(test, ["noul_z"], np.array([fill_z]))
    platt = fit_logistic(Zc, [r["exists"] for r in cal])
    pe_cal, pe_test = platt(Zc), platt(Zt)

    cal_pos = [r for r in cal if r["exists"]]
    fill_f = np.nanmedian(np.array([[r[k] if r[k] is not None else np.nan for k in FEATURES] for r in cal_pos], float), 0)
    Xc, nf_c = matrix(cal_pos, FEATURES, fill_f)
    box_model = fit_logistic(Xc, [r["iou"] >= 0.5 for r in cal_pos])

    pos = [r for r in test if r["exists"]]
    neg = [r for r in test if not r["exists"]]
    Xp, nf_t = matrix(pos, FEATURES, fill_f)
    correct = np.array([r["iou"] >= 0.5 for r in pos], float)
    pb_pos = box_model(Xp)

    def end_to_end(th, rs, pe):
        X, _ = matrix(rs, FEATURES, fill_f)
        pb = box_model(X)
        ok, conf = [], []
        for r, e, b in zip(rs, pe, pb):
            if e < th:
                ok.append(not r["exists"])
                conf.append(1 - e)
            else:
                ok.append(r["exists"] and r["iou"] >= 0.5)
                conf.append(e * b)
        return np.array(ok, float), np.array(conf)

    tuned = float(max(np.linspace(0.05, 0.95, 37), key=lambda th: end_to_end(th, cal, pe_cal)[0].mean()))
    single = {k: auroc([-x if k == "mean_entropy" else x for x in Xp[:, i]], correct.astype(int))
              for i, k in enumerate(FEATURES)}
    res = {
        "imputed_items": {"noul_cal": nz_c, "noul_test": nz_t, "box_cal": nf_c, "box_test": nf_t},
        "present": {"n": len(pos), "acc_iou0.5": round(float(correct.mean()), 4),
                    "mean_iou": round(float(np.mean([r["iou"] for r in pos])), 4),
                    "box_conf_ece_raw_mean_p": ece(Xp[:, 0], correct),
                    "box_conf_ece_calibrated": ece(pb_pos, correct),
                    "box_conf_auroc": auroc(pb_pos, correct.astype(int)),
                    "single_feature_auroc": single,
                    "acc_at_coverage_0.8": coverage_acc(pb_pos, correct, 0.8),
                    "acc_at_coverage_0.5": coverage_acc(pb_pos, correct, 0.5)},
        "existence": {"auroc": auroc(pe_test, [int(r["exists"]) for r in test]),
                      "ece": ece(pe_test, [r["exists"] for r in test])},
        "box_only_accuracy_if_never_abstaining": round(float(np.mean([r["exists"] and r["iou"] >= 0.5 for r in test])), 4),
    }
    for label, th in (("threshold_0.5", 0.5), ("threshold_tuned", tuned)):
        ok, conf = end_to_end(th, test, pe_test)
        res[f"end_to_end_{label}"] = {
            "threshold": round(th, 3), "accuracy": round(float(ok.mean()), 4), "decision_conf_ece": ece(conf, ok),
            "acc_at_coverage_0.8": coverage_acc(conf, ok, 0.8),
            "abstain_rate_absent": round(float(np.mean([e < th for e, r in zip(pe_test, test) if not r["exists"]])), 4) if neg else None,
            "wrongful_abstain_rate_present": round(float(np.mean([e < th for e, r in zip(pe_test, test) if r["exists"]])), 4)}
    print(json.dumps(res, indent=1))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
