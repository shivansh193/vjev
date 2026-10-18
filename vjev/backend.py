"""Model backends. A backend exposes two primitives; everything else is built on them:

    next_token_logits(image, prompt_text, token_strings) -> np.ndarray   # one forward pass
    ground(image, target) -> (box | None, stats)                           # short generation

`QwenVLBackend` wraps Qwen2.5-VL (transformers). Tests use a fake backend, so the
decision logic can be exercised without a GPU or model weights.
"""
import base64
import io
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PREFILL = '```json\n[\n\t{"bbox_2d": ['
Box4 = List[float]                     # [left, top, right, bottom], normalised to [0, 1]


def load_image(asset):
    """PIL image from a PIL image, a file path, raw bytes or a base64 string."""
    from PIL import Image
    if asset is None:
        return None
    if hasattr(asset, "convert"):
        return asset.convert("RGB")
    if isinstance(asset, (bytes, bytearray)):
        return Image.open(io.BytesIO(asset)).convert("RGB")
    if isinstance(asset, str):
        if asset.startswith("data:"):
            asset = asset.split(",", 1)[1]
        try:
            return Image.open(asset).convert("RGB")
        except (OSError, ValueError):
            return Image.open(io.BytesIO(base64.b64decode(asset))).convert("RGB")
    raise TypeError(f"unsupported image asset: {type(asset).__name__}")


class QwenVLBackend:
    def __init__(self, model: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: Optional[str] = None,
                 pixels: int = 448 * 448):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoProcessor.from_pretrained(model, min_pixels=pixels, max_pixels=pixels)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, torch_dtype=dtype).to(self.device).eval()
        self._ids: Dict[str, int] = {}

    def token_id(self, s: str) -> int:
        if s not in self._ids:
            self._ids[s] = self.proc.tokenizer.encode(s, add_special_tokens=False)[0]
        return self._ids[s]

    def _inputs(self, image, text: str, prefill: str = ""):
        content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": text}]
        prompt = self.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True)
        kw = {"images": [image]} if image is not None else {}
        return self.proc(text=[prompt + prefill], return_tensors="pt", **kw).to(self.device)

    def next_token_logits(self, image, text: str, tokens: Sequence[str]) -> np.ndarray:
        with self.torch.no_grad():
            logits = self.model(**self._inputs(image, text)).logits[0, -1].float()
        return logits[[self.token_id(t) for t in tokens]].cpu().numpy()

    def ground(self, image, target: str) -> Tuple[Optional[Box4], dict]:
        torch = self.torch
        inp = self._inputs(image, f"Locate \"{target}\" in the image and output its bbox_2d coordinates in JSON.", PREFILL)
        with torch.no_grad():
            g = self.model.generate(**inp, max_new_tokens=24, do_sample=False, output_scores=True,
                                    return_dict_in_generate=True)
        new = g.sequences[0, inp["input_ids"].shape[1]:]
        text, logps, ents = "", [], []
        for s, t in zip(g.scores, new):
            lp = torch.log_softmax(s[0].float(), -1)
            logps.append(lp[t].item())
            p = lp.exp()
            ents.append(float(-(p[p > 0] * lp[p > 0]).sum()))
            text += self.proc.tokenizer.decode([int(t)], skip_special_tokens=True)
            if "]" in text:
                break
        stats = {"mean_p": float(np.exp(np.mean(logps))) if logps else None,
                 "min_p": float(np.exp(min(logps))) if logps else None,
                 "mean_entropy": float(np.mean(ents)) if ents else None}
        nums = re.findall(r"-?\d+\.?\d*", text.split("]")[0])
        if len(nums) < 4:
            return None, stats
        _, gh, gw = inp["image_grid_thw"][0].tolist()
        W, H = gw * 14, gh * 14                      # coordinates are in the resized image
        x0, y0, x1, y1 = [float(v) for v in nums[:4]]
        b = [x0 / W, y0 / H, x1 / W, y1 / H]
        return [min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])], stats
