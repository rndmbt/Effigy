"""Learned albedo/roughness/metallic from Marigold IID Appearance. Predictions are estimates, not measured
material properties. Model renders are decomposed in view space and projected into the existing UVs;
never feed an atlas of unrelated charts to the estimator. Existing AO and tangent normals stay untouched.
"""

import gc
import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image

from steps._common import StepError

MODEL = "prs-eth/marigold-iid-appearance-v1-1"
REVISION = "e7280a0a0fc5a0df0b36050882b3d8b77da22fd9"


class Estimator:
    def __init__(self, seed: int = 42):
        import torch
        from diffusers import MarigoldIntrinsicsPipeline
        from huggingface_hub import snapshot_download

        root = Path(os.environ.get("EFFIGY_MODELS", Path(__file__).resolve().parents[1] / "models"))
        cache = Path(
            os.environ.get("EFFIGY_PBR_CACHE") or os.environ.get("HF_HUB_CACHE", root / "hub")
        )
        try:
            path = snapshot_download(
                MODEL,
                revision=REVISION,
                cache_dir=cache,
                allow_patterns=["*.json", "tokenizer/*", "*/*.fp16.safetensors"],
                local_files_only=True,
            )
        except OSError as e:
            raise StepError(
                "missing_input",
                "Marigold IID Appearance weights missing: install Learned PBR in Setup",
            ) from e
        self.pipe = MarigoldIntrinsicsPipeline.from_pretrained(
            path, variant="fp16", torch_dtype=torch.float16, local_files_only=True
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.seed = seed
        self.cache = {}
        self.role = "basecolor"
        self.models = [{"id": MODEL, "revision": REVISION}]

    def predict(self, image: Image.Image) -> dict:
        import torch

        rgb = image.convert("RGB")
        key = hashlib.sha256(rgb.tobytes()).hexdigest()
        if key not in self.cache:
            with torch.inference_mode():
                result = self.pipe(
                    rgb,
                    num_inference_steps=10,
                    ensemble_size=1,
                    processing_resolution=768,
                    match_input_resolution=True,
                    generator=torch.Generator("cuda").manual_seed(self.seed),
                ).prediction
            # Raw predictions: albedo is sRGB; material R = linear roughness, G = linear metallic.
            # The visualisation helper gamma-corrects scalar maps; do NOT use it for material data.
            a = np.clip(result, 0, 1)
            self.cache[key] = {
                "basecolor": Image.fromarray((a[0] * 255).round().astype(np.uint8)),
                "roughness": Image.fromarray((a[1, ..., 0] * 255).round().astype(np.uint8)),
                "metallic": Image.fromarray((a[1, ..., 1] * 255).round().astype(np.uint8)),
            }
        return self.cache[key]

    def encode(self, prompts):
        return {p: p for p in prompts}

    def __call__(self, ctrl, prompt, strength, scale, seed):
        return np.asarray(self.predict(Image.fromarray(ctrl))[self.role].convert("RGB"))

    def close(self):
        import torch

        del self.pipe
        self.cache.clear()
        gc.collect()
        torch.cuda.empty_cache()


def estimate(src: Path, out: Path, maps: dict, model: bool, params: dict) -> tuple[dict, dict]:
    from steps import refine

    estimator = Estimator(int(params.get("seed", 42)))
    try:
        if not model:
            return estimator.predict(maps["basecolor"]), {"method": MODEL}
        generated, reports = {}, {}
        # All three passes render the SAME original colour, so one inference serves all channels per view.
        for role in ("basecolor", "roughness", "metallic"):
            estimator.role = role
            work = out / ("_pbr_" + role)
            work.mkdir()
            result = refine.run(
                "run",
                src,
                work,
                {
                    "_estimator": estimator,
                    "strength": 1,
                    "face_strength": 1,
                    "coverage": 0,
                    "zoom": 0.5,
                    "_material_role": role,
                    "_fallback": maps.get(role),
                    "seed": int(params.get("seed", 42)),
                },
            )
            image = Image.open(work / "basecolor.png").copy()
            generated[role] = image.convert("RGB" if role == "basecolor" else "L")
            reports[role] = {
                "views": result["metrics"]["views"],
                "seen": result["metrics"]["refined"],
            }
        return generated, {"method": MODEL, "projection": reports}
    finally:
        estimator.close()
        import shutil

        for role in ("basecolor", "roughness", "metallic"):
            shutil.rmtree(out / ("_pbr_" + role), ignore_errors=True)
