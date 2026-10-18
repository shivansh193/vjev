"""Jev-style Box answer, step 2: a calibrated, abstaining box head on a frozen VLM.

Jev 1.14 returns one box per question with a confidence "derived from the answer
distribution". No open recreation has a Box answer. This tests one design:

Backbone  Qwen2.5-VL-3B-Instruct, frozen, fp16. Prompt = expression FIRST, then the image,
          then "Answer:", so every image token is conditioned on what we look for (causal LM)
          and the final token sees everything. We keep last-layer states of the image tokens
          (K, one per 28x28 px cell) and of the final token (q).
Head      pointer   : softmax over image tokens of (W_q q) . (W_k k_i + pos_i)  -> where-map
          edges     : 4 categorical distributions (left, top, right, bottom) over 64 bins,
                      from [q, pointer-pooled features, pointer mean/spread]
          existence : P(target is in the image) (Noul), trained with 25% negatives
                      (expression taken from another image)
Box       expected value of each edge distribution
Conf      sharpness = product over edges of the probability mass within +-1 bin of the
          expected bin; box confidence = P(exists) * calibrated P(IoU>=0.5 | sharpness)
          (histogram calibration fitted on a held-out split)
Data      lmms-lab/RefCOCO: head trained on `val` (minus 1000 for calibration);
          evaluation on testA + testB
Baseline  the same VLM generating its own box as text (native grounding), on 500 items,
          confidence = mean token probability of the generated answer

Metrics   Acc@IoU0.5 and mean IoU (targets present); existence AUROC / accuracy; ECE of box
          confidence vs IoU>=0.5 (raw and calibrated); accuracy at 50% / 80% coverage
          (keep the most confident boxes); abstention rate on absent targets

Outputs in /kaggle/working: vjev_box.json, box_head.pt
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
import torch.nn as nn
import torch.nn.functional as F

if os.path.exists("/kaggle"):                       # only install on Kaggle, never into a local env
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.49", "accelerate", "datasets"],
                   check=True)
from datasets import load_dataset  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402

MODEL = os.environ.get("VB_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
OUT = os.environ.get("VB_OUT", "/kaggle/working")
TMP = os.environ.get("VB_TMP", "/kaggle/tmp/vjev_box")
N_TRAIN = int(os.environ.get("VB_N_TRAIN", 7800))
N_CAL = int(os.environ.get("VB_N_CAL", 1000))
N_EVAL = int(os.environ.get("VB_N_EVAL", 0))          # 0 = all of testA + testB
N_BASELINE = int(os.environ.get("VB_N_BASELINE", 500))
NEG_RATE = 0.25
PIX = 448 * 448
MAXTOK = 330                                  # 448x448 -> ~16x16 tokens; room for odd aspect ratios
PROJ = 768                                    # fixed random projection of the 2048-d states (fits in RAM)
BINS = 64
EPOCHS = int(os.environ.get("VB_EPOCHS", 25))
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------- data

def norm_item(ex):
    """-> (PIL image, expression, [x0, y0, x1, y1] normalised) or None."""
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
    x, y, bw, bh = [float(v) for v in bb]                 # COCO xywh in pixels
    box = [x / w, y / h, (x + bw) / w, (y + bh) / h]
    if not (0 <= box[0] < box[2] <= 1.001 and 0 <= box[1] < box[3] <= 1.001):
        return None
    return img, str(expr).strip(), [min(max(v, 0.0), 1.0) for v in box]


def load_splits():
    rng = random.Random(SEED)
    val = load_dataset("lmms-lab/RefCOCO", split="val")
    print("RefCOCO columns:", val.column_names, flush=True)
    ok = [norm_item(val[i]) for i in range(5)]
    if not any(ok):
        raise SystemExit(f"cannot parse RefCOCO items; first item keys/values: "
                         f"{ {k: str(v)[:80] for k, v in val[0].items() if k != 'image'} }")
    print("example:", ok[0][1], [round(v, 3) for v in ok[0][2]] if ok[0] else None, flush=True)
    idx = list(range(len(val)))
    rng.shuffle(idx)
    train_src = [(val, i) for i in idx[:N_TRAIN]]
    cal_src = [(val, i) for i in idx[N_TRAIN:N_TRAIN + N_CAL]]
    test_src = []
    for split in ("testA", "testB"):
        ds = load_dataset("lmms-lab/RefCOCO", split=split)
        test_src += [(ds, i) for i in range(len(ds))]
    rng.shuffle(test_src)
    if N_EVAL:
        test_src = test_src[:N_EVAL]
    return train_src, cal_src, test_src


def with_negatives(src, rng):
    """Items: (ds, i, expr_from=(ds, j) or None). Negatives borrow another image's expression."""
    items = [(ds, i, None) for ds, i in src]
    n_neg = int(len(items) * NEG_RATE)
    for _ in range(n_neg):
        ds, i = rng.choice(src)
        dj, j = rng.choice(src)
        if dj is ds and j == i:
            continue
        items.append((ds, i, (dj, j)))
    rng.shuffle(items)
    return items


# ---------------------------------------------------------------- backbone features

class Backbone:
    def __init__(self):
        self.proc = AutoProcessor.from_pretrained(MODEL, min_pixels=PIX, max_pixels=PIX)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
        cfg = self.model.config
        text_cfg = getattr(cfg, "text_config", None)
        # newer transformers keep these on text_config; older ones on the top-level config
        self.image_token_id = getattr(cfg, "image_token_id", None) or getattr(text_cfg, "image_token_id", None)
        hidden = getattr(cfg, "hidden_size", None) or text_cfg.hidden_size
        g = torch.Generator().manual_seed(SEED)
        self.P = (torch.randn(hidden, PROJ, generator=g) / math.sqrt(PROJ)).to(DEV)   # fixed, roughly norm-preserving
        self.dim = PROJ

    def prompt(self, expr):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": f"Find this in the image: {expr}"},
            {"type": "image"},
            {"type": "text", "text": "Where is it?"}]}]
        return self.proc.apply_chat_template(msgs, add_generation_prompt=True)

    @torch.no_grad()
    def features(self, image, expr):
        inp = self.proc(text=[self.prompt(expr)], images=[image], return_tensors="pt").to(DEV)
        out = self.model(**inp, output_hidden_states=True)
        hs = out.hidden_states[-1][0].float()
        # a few hidden dims carry huge activations; RMS-normalise each vector so the projected
        # features stay O(1) and cannot overflow fp16 storage
        hs = hs / hs.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
        hs = hs @ self.P
        pos = (inp["input_ids"][0] == self.image_token_id).nonzero()[:, 0]
        _, gh, gw = inp["image_grid_thw"][0].tolist()
        th, tw = gh // 2, gw // 2                          # 2x2 patch merge -> one token per 28 px
        k = hs[pos].float().cpu().numpy()
        assert len(k) == th * tw, (len(k), th, tw)
        return hs[-1].float().cpu().numpy(), k, (th, tw)

    @torch.no_grad()
    def generate_box(self, image, expr):
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": f"Locate \"{expr}\" in the image and output its bbox_2d coordinates in JSON."}]}]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = self.proc(text=[prompt], images=[image], return_tensors="pt").to(DEV)
        g = self.model.generate(**inp, max_new_tokens=60, do_sample=False, output_scores=True,
                                return_dict_in_generate=True)
        new = g.sequences[0, inp["input_ids"].shape[1]:]
        text = self.proc.tokenizer.decode(new, skip_special_tokens=True)
        logp = [torch.log_softmax(s[0].float(), -1)[t].item() for s, t in zip(g.scores, new)]
        _, gh, gw = inp["image_grid_thw"][0].tolist()
        m = re.search(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]", text)
        nums = [float(v) for v in m.groups()] if m else []
        if len(nums) < 4:
            return None, float(np.exp(np.mean(logp))) if logp else 0.0
        W, H = gw * 14, gh * 14                            # coordinates are in the resized image
        x0, y0, x1, y1 = nums
        return [x0 / W, y0 / H, x1 / W, y1 / H], float(np.exp(np.mean(logp)))


def extract(bb, items, name):
    """Writes q (N, D), K (N, MAXTOK, D) fp16 memmaps and meta; returns meta list."""
    os.makedirs(TMP, exist_ok=True)
    n = len(items)
    q = np.lib.format.open_memmap(f"{TMP}/{name}_q.npy", "w+", np.float16, (n, bb.dim))
    K = np.lib.format.open_memmap(f"{TMP}/{name}_k.npy", "w+", np.float16, (n, MAXTOK, bb.dim))
    meta = []
    t = time.time()
    for r, (ds, i, other) in enumerate(items):
        got = norm_item(ds[i])
        expr_src = norm_item(other[0][other[1]]) if other else got
        if got is None or expr_src is None:
            meta.append(None)
            continue
        img, _, box = got
        expr = expr_src[1]
        qv, kv, (th, tw) = bb.features(img, expr)
        ntok = min(len(kv), MAXTOK)
        q[r] = qv
        K[r, :ntok] = kv[:ntok]
        meta.append({"box": box, "exists": other is None, "th": th, "tw": tw, "ntok": ntok, "expr": expr})
        if r % 500 == 0:
            print(f"  {name} {r}/{n}  {(time.time() - t) / max(r, 1):.2f}s/item  {(time.time() - T0) / 60:.0f} min",
                  flush=True)
    q.flush()
    K.flush()
    return meta


# ---------------------------------------------------------------- head

def token_xy(th, tw):
    ys, xs = np.meshgrid((np.arange(th) + 0.5) / th, (np.arange(tw) + 0.5) / tw, indexing="ij")
    return np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)


class BoxHead(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.h = h
        self.qp, self.kp, self.vp = nn.Linear(d, h), nn.Linear(d, h), nn.Linear(d, h)
        self.pos = nn.Sequential(nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, h))
        self.edge = nn.Sequential(nn.Linear(2 * h + 4, 512), nn.ReLU(), nn.Dropout(0.1), nn.Linear(512, 4 * BINS))
        self.exist = nn.Sequential(nn.Linear(2 * h, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, q, K, xy, mask):
        qh = self.qp(q)                                               # B, h
        kh = self.kp(K) + self.pos(xy)                                # B, N, h
        logits = (kh @ qh[..., None])[..., 0] / math.sqrt(self.h)
        logits = logits.masked_fill(~mask, -1e4)
        att = torch.softmax(logits, -1)                               # B, N
        pooled = (att[..., None] * self.vp(K)).sum(1)
        mu = (att[..., None] * xy).sum(1)
        sd = ((att[..., None] * (xy - mu[:, None]) ** 2).sum(1) + 1e-6).sqrt()
        edges = self.edge(torch.cat([qh, pooled, mu, sd], -1)).view(-1, 4, BINS)
        exist = self.exist(torch.cat([qh, pooled], -1))[:, 0]
        return logits, edges, exist


def batch(qm, km, meta, rows):
    q = torch.from_numpy(np.nan_to_num(qm[rows].astype(np.float32), posinf=0.0, neginf=0.0))
    K = torch.from_numpy(np.nan_to_num(km[rows].astype(np.float32), posinf=0.0, neginf=0.0))
    xy = np.zeros((len(rows), MAXTOK, 2), np.float32)
    mask = np.zeros((len(rows), MAXTOK), bool)
    box = np.zeros((len(rows), 4), np.float32)
    ex = np.zeros(len(rows), np.float32)
    for b, r in enumerate(rows):
        m = meta[r]
        g = token_xy(m["th"], m["tw"])[:m["ntok"]]
        xy[b, :len(g)] = g
        mask[b, :len(g)] = True
        box[b] = m["box"]
        ex[b] = float(m["exists"])
    return q, K, torch.from_numpy(xy), torch.from_numpy(mask), torch.from_numpy(box), torch.from_numpy(ex)


def losses(head, q, K, xy, mask, box, ex):
    logits, edges, exist = head(q, K, xy, mask)
    l_ex = F.binary_cross_entropy_with_logits(exist, ex)
    pos = ex > 0.5
    if not pos.any():
        return l_ex, l_ex.detach() * 0, l_ex.detach() * 0
    # pointer target: tokens whose centre lies in the box (uniform), else the nearest token
    inside = ((xy[..., 0] >= box[:, None, 0]) & (xy[..., 0] <= box[:, None, 2]) &
              (xy[..., 1] >= box[:, None, 1]) & (xy[..., 1] <= box[:, None, 3]) & mask).float()
    centre = torch.stack([(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2], -1)
    d = ((xy - centre[:, None]) ** 2).sum(-1).masked_fill(~mask, 1e9)
    nearest = F.one_hot(d.argmin(1), MAXTOK).float()
    tgt = torch.where(inside.sum(1, keepdim=True) > 0, inside, nearest)
    tgt = tgt / tgt.sum(1, keepdim=True)
    l_ptr = -(tgt * F.log_softmax(logits, -1)).sum(-1)[pos].mean()
    # edges: Gaussian soft labels (sigma = 1 bin) on left, top, right, bottom
    centres = (torch.arange(BINS, dtype=torch.float32, device=box.device) + 0.5) / BINS
    soft = torch.exp(-((centres[None, None] - box[..., None]) * BINS) ** 2 / 2)
    soft = soft / soft.sum(-1, keepdim=True)
    l_edge = -(soft * F.log_softmax(edges, -1)).sum(-1).mean(-1)[pos].mean()
    return l_ex, l_ptr, l_edge


def predict(head, qm, km, meta, rows, temp=1.0):
    head.eval()
    out = []
    centres = (torch.arange(BINS, dtype=torch.float32) + 0.5) / BINS
    with torch.no_grad():
        for s in range(0, len(rows), 256):
            r = rows[s:s + 256]
            q, K, xy, mask, _, _ = batch(qm, km, meta, r)
            _, edges, exist = head(q.to(DEV), K.to(DEV), xy.to(DEV), mask.to(DEV))
            p = torch.softmax(edges.float().cpu() / temp, -1)              # B, 4, BINS
            exp_v = (p * centres).sum(-1)                                   # expected edge
            k = (exp_v * BINS).long().clamp(0, BINS - 1)
            idx = torch.stack([(k + o).clamp(0, BINS - 1) for o in (-1, 0, 1)], -1)
            near = p.gather(-1, idx).sum(-1)                                # mass within +-1 bin
            sharp = near.prod(-1)
            pe = torch.sigmoid(exist.float().cpu())
            for b in range(len(r)):
                x0, y0, x1, y1 = exp_v[b].tolist()
                out.append({"box": [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)],
                            "sharp": float(sharp[b]), "p_exist": float(pe[b])})
    return out


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
    """Monotone (cumulative-max) histogram calibration on equal-count bins."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    qs = np.quantile(conf, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    vals = []
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (conf > lo) & (conf <= hi)
        vals.append(correct[m].mean() if m.any() else (vals[-1] if vals else 0.0))
    vals = np.maximum.accumulate(np.array(vals))
    return lambda c: vals[np.clip(np.searchsorted(qs, np.asarray(c, float), side="left") - 1, 0, bins - 1)]


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

def main():
    rng = random.Random(SEED)
    train_src, cal_src, test_src = load_splits()
    sets = {"train": with_negatives(train_src, rng), "cal": with_negatives(cal_src, rng),
            "test": with_negatives(test_src, rng)}
    print({k: len(v) for k, v in sets.items()}, flush=True)
    bb = Backbone()

    # native-generation baseline (targets present only)
    base = []
    tb = time.time()
    for ds, i in test_src[:N_BASELINE]:
        got = norm_item(ds[i])
        if got is None:
            continue
        img, expr, box = got
        pred, conf = bb.generate_box(img, expr)
        base.append({"iou": iou(pred, box) if pred else 0.0, "conf": conf, "parsed": pred is not None})
    base_secs = (time.time() - tb) / max(len(base), 1)
    print(f"baseline done: {len(base)} items, {base_secs:.2f}s/item", flush=True)

    data = {}
    for name, items in sets.items():
        meta = extract(bb, items, name)
        keep = [r for r, m in enumerate(meta) if m is not None]
        if not keep:
            raise SystemExit(f"no usable items in {name}")
        data[name] = (np.load(f"{TMP}/{name}_q.npy"), np.load(f"{TMP}/{name}_k.npy"), meta, keep)   # in RAM
        qf = data[name][0][keep]
        print(f"  {name}: non-finite q share {float((~np.isfinite(qf)).mean()):.2e}  "
              f"|q| mean {float(np.abs(qf).mean()):.3f}", flush=True)
    feat_secs = (time.time() - tb) / sum(len(v) for v in sets.values())
    del bb
    torch.cuda.empty_cache()

    # train the head
    qm, km, meta, rows = data["train"]
    head = BoxHead(qm.shape[1]).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = EPOCHS * math.ceil(len(rows) / 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.1)
    for ep in range(EPOCHS):
        head.train()
        random.shuffle(rows)
        tot = np.zeros(3)
        for s in range(0, len(rows), 64):
            q, K, xy, mask, box, ex = batch(qm, km, meta, rows[s:s + 64])
            parts = losses(head, *(t.to(DEV) for t in (q, K, xy, mask, box, ex)))
            loss = sum(parts)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += [p.item() for p in parts]
        if ep % 5 == 4 or ep == EPOCHS - 1:
            print(f"epoch {ep + 1}  exist {tot[0]:.1f}  pointer {tot[1]:.1f}  edges {tot[2]:.1f}", flush=True)
    torch.save(head.state_dict(), f"{OUT}/box_head.pt")

    def evaluate(name):
        qm_, km_, meta_, rows_ = data[name]
        pr = predict(head, qm_, km_, meta_, rows_)
        recs = []
        for r, p in zip(rows_, pr):
            m = meta_[r]
            recs.append({"exists": m["exists"], "iou": iou(p["box"], m["box"]) if m["exists"] else 0.0, **p})
        return recs

    cal, test = evaluate("cal"), evaluate("test")
    cal_pos = [r for r in cal if r["exists"]]
    calib = fit_hist([r["sharp"] for r in cal_pos], [r["iou"] >= 0.5 for r in cal_pos])
    pos = [r for r in test if r["exists"]]
    neg = [r for r in test if not r["exists"]]
    correct = np.array([r["iou"] >= 0.5 for r in pos], float)
    raw_conf = np.array([r["sharp"] for r in pos])
    cal_conf = calib(raw_conf)
    full_conf = cal_conf * np.array([r["p_exist"] for r in pos])

    head_res = {
        "acc_iou0.5": round(float(correct.mean()), 4),
        "mean_iou": round(float(np.mean([r["iou"] for r in pos])), 4),
        "box_conf_ece_raw": ece(raw_conf, correct),
        "box_conf_ece_calibrated": ece(cal_conf, correct),
        "acc_at_coverage_0.8": coverage_acc(cal_conf, correct, 0.8),
        "acc_at_coverage_0.5": coverage_acc(cal_conf, correct, 0.5),
        "existence_auroc": auroc([r["p_exist"] for r in test], [int(r["exists"]) for r in test]),
        "existence_acc@0.5": round(float(np.mean([(r["p_exist"] >= 0.5) == r["exists"] for r in test])), 4),
        "abstain_rate_absent": round(float(np.mean([r["p_exist"] < 0.5 for r in neg])), 4) if neg else None,
        "false_box_rate_present": round(float(np.mean([r["p_exist"] < 0.5 for r in pos])), 4),
        "decision_conf_ece": ece(full_conf, correct),
        "n_present": len(pos), "n_absent": len(neg),
    }
    b_correct = np.array([r["iou"] >= 0.5 for r in base], float)
    b_conf = np.array([r["conf"] for r in base])
    base_res = {"acc_iou0.5": round(float(b_correct.mean()), 4) if len(base) else None,
                "mean_iou": round(float(np.mean([r["iou"] for r in base])), 4) if base else None,
                "parse_rate": round(float(np.mean([r["parsed"] for r in base])), 4) if base else None,
                "conf_ece": ece(b_conf, b_correct) if len(base) else None,
                "acc_at_coverage_0.5": coverage_acc(b_conf, b_correct, 0.5) if len(base) else None,
                "seconds_per_item": round(base_secs, 2), "n": len(base)}

    out = {"model": MODEL, "data": "lmms-lab/RefCOCO (head: val; eval: testA+testB)",
           "n": {k: len(v[3]) for k, v in data.items()}, "neg_rate": NEG_RATE, "bins": BINS,
           "box_head": head_res, "native_generation_baseline": base_res,
           "feature_seconds_per_item": round(feat_secs, 3), "total_minutes": round((time.time() - T0) / 60, 1)}
    json.dump(out, open(f"{OUT}/vjev_box.json", "w"), indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
