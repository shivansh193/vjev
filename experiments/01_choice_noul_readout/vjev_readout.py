"""Jev-style typed decisions on images, step 1: training-free option readout + calibration.

Question: can a small open vision-language model give typed, calibrated answers to
image questions by READING option-token probabilities instead of generating text?
(The AnyJev / imajev recipe, applied to images.)

Model    Qwen2.5-VL-3B-Instruct, frozen, fp16
Data     Oxford-IIIT Pets (37 breeds), test split; first 300 images = calibration split,
         next N images = evaluation split
Choice   "Which breed is shown?" with K=5 candidates (truth + 4 random distractors),
         labelled A-E; answer = softmax over the A-E token logits at the answer position
Noul     "Is this a {breed}?" (half true, half a random wrong breed); P(yes) from the
         Yes / No token logits
Readouts raw      one prompt, fixed option order
         rotated  mean probability over all K cyclic rotations of the option list
                  (removes position bias)
         rot+T    rotated, then one temperature fitted on the calibration split (NLL)
Metrics  accuracy, ECE (15 bins), Brier, NLL, seconds per question

Outputs in /kaggle/working: vjev_readout.json
"""
import json
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.49", "accelerate", "qwen-vl-utils"],
               check=True)
import torchvision  # noqa: E402
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402

MODEL = os.environ.get("VJ_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
OUT = os.environ.get("VJ_OUT", "/kaggle/working")
N_CAL = int(os.environ.get("VJ_N_CAL", 300))
N_EVAL = int(os.environ.get("VJ_N_EVAL", 1000))
K = 5
LETTERS = "ABCDE"
SEED = 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED)
np.random.seed(SEED)


def load():
    ds = torchvision.datasets.OxfordIIITPet("/kaggle/tmp/pets", split="test", download=True)
    names = [c.replace("_", " ") for c in ds.classes]
    order = list(range(len(ds)))
    random.Random(SEED).shuffle(order)
    return ds, names, order[:N_CAL], order[N_CAL:N_CAL + N_EVAL]


class Reader:
    def __init__(self):
        self.proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28, max_pixels=448 * 448)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL, torch_dtype=torch.float16 if DEV == "cuda" else torch.float32).to(DEV).eval()
        tok = self.proc.tokenizer
        ids = lambda words: [tok.encode(w, add_special_tokens=False)[0] for w in words]
        self.letter_ids = ids(list(LETTERS))
        self.yes_no_ids = ids(["Yes", "No"])

    @torch.no_grad()
    def next_logits(self, image, text):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inp = self.proc(text=[prompt], images=[image], return_tensors="pt").to(DEV)
        return self.model(**inp).logits[0, -1].float()

    def choice(self, image, options):
        text = ("Which breed of pet is shown in the image?\nOptions:\n"
                + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
                + "\nAnswer with the letter only.")
        return self.next_logits(image, text)[self.letter_ids].cpu().numpy()

    def noul(self, image, breed):
        text = f"Is the pet in this image a {breed}? Answer Yes or No."
        z = self.next_logits(image, text)[self.yes_no_ids].cpu().numpy()
        return z[0] - z[1]                                   # logit of P(yes) vs P(no)


def softmax(z, t=1.0):
    z = np.asarray(z, float) / t
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def metrics(probs, correct_idx):
    """probs: (N, K) over options; correct_idx: (N,)."""
    probs = np.asarray(probs, float)
    pred = probs.argmax(1)
    conf = probs.max(1)
    acc = (pred == correct_idx)
    bins = np.linspace(0, 1, 16)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    onehot = np.eye(probs.shape[1])[correct_idx]
    return {"accuracy": round(float(acc.mean()), 4), "ece": round(float(ece), 4),
            "brier": round(float(((probs - onehot) ** 2).sum(1).mean()), 4),
            "nll": round(float(-np.log(np.clip(probs[np.arange(len(probs)), correct_idx], 1e-12, 1)).mean()), 4)}


def fit_temperature(logit_rows, correct_idx):
    """One temperature minimising NLL; applied to log-probabilities (never changes the argmax)."""
    grid = np.exp(np.linspace(np.log(0.2), np.log(5), 80))
    nll = lambda t: -np.mean([np.log(max(softmax(z, t)[c], 1e-12)) for z, c in zip(logit_rows, correct_idx)])
    return float(min(grid, key=nll))


def run_choice(reader, ds, names, idxs):
    """Returns raw probs, rotated log-probs and correct option index per item."""
    raw, rot_logp, correct = [], [], []
    rng = random.Random(SEED + 1)
    for n, i in enumerate(idxs):
        image, label = ds[i]
        distract = rng.sample([c for c in range(len(names)) if c != label], K - 1)
        opts = [label] + distract
        rng.shuffle(opts)
        c = opts.index(label)
        per_option = np.zeros(K)
        for r in range(K):                                   # cyclic rotations of the option list
            rotated = opts[r:] + opts[:r]
            p = softmax(reader.choice(image, [names[o] for o in rotated]))
            if r == 0:
                raw.append(p)
            for pos, o in enumerate(rotated):
                per_option[opts.index(o)] += np.log(max(p[pos], 1e-12)) / K
        rot_logp.append(per_option)                          # geometric mean over rotations
        correct.append(c)
        if n % 100 == 0:
            print(f"  choice {n}/{len(idxs)}", flush=True)
    return np.array(raw), np.array(rot_logp), np.array(correct)


def run_noul(reader, ds, names, idxs):
    z, y = [], []
    rng = random.Random(SEED + 2)
    for n, i in enumerate(idxs):
        image, label = ds[i]
        truth = rng.random() < 0.5
        breed = names[label] if truth else names[rng.choice([c for c in range(len(names)) if c != label])]
        z.append(reader.noul(image, breed))
        y.append(int(truth))
        if n % 200 == 0:
            print(f"  noul {n}/{len(idxs)}", flush=True)
    return np.array(z), np.array(y)


def main():
    t0 = time.time()
    ds, names, cal_idx, eval_idx = load()
    reader = Reader()
    print(f"{MODEL} on {DEV}; cal {len(cal_idx)}  eval {len(eval_idx)}  load {time.time() - t0:.0f}s", flush=True)

    tc = time.time()
    _, cal_rot, cal_c = run_choice(reader, ds, names, cal_idx)
    raw, rot, c = run_choice(reader, ds, names, eval_idx)
    choice_secs = (time.time() - tc) / ((len(cal_idx) + len(eval_idx)) * K)
    t_choice = fit_temperature(cal_rot, cal_c)
    choice = {
        "raw": metrics(raw, c),
        "rotated": metrics(softmax(rot), c),
        "rotated+T": metrics(softmax(rot, t_choice), c),
        "temperature": round(t_choice, 3),
        "seconds_per_forward": round(choice_secs, 3),
        "rotations_per_question": K,
    }
    print("choice", json.dumps(choice), flush=True)

    tn = time.time()
    cal_z, cal_y = run_noul(reader, ds, names, cal_idx)
    z, y = run_noul(reader, ds, names, eval_idx)
    noul_secs = (time.time() - tn) / (len(cal_idx) + len(eval_idx))
    two = lambda logit, t=1.0: softmax(np.stack([logit, np.zeros_like(logit)], 1), t)
    t_noul = fit_temperature(np.stack([cal_z, np.zeros_like(cal_z)], 1), 1 - cal_y)
    noul = {
        "raw": metrics(two(z), 1 - y),
        "raw+T": metrics(two(z, t_noul), 1 - y),
        "temperature": round(t_noul, 3),
        "p_yes_mean_true": round(float(two(z)[y == 1, 0].mean()), 3),
        "p_yes_mean_false": round(float(two(z)[y == 0, 0].mean()), 3),
        "seconds_per_question": round(noul_secs, 3),
    }
    print("noul", json.dumps(noul), flush=True)

    out = {"model": MODEL, "dataset": "Oxford-IIIT Pets (test)", "n_cal": len(cal_idx), "n_eval": len(eval_idx),
           "k_options": K, "choice": choice, "noul": noul, "total_minutes": round((time.time() - t0) / 60, 1)}
    json.dump(out, open(f"{OUT}/vjev_readout.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
