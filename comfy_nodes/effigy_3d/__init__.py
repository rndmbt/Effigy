# SPDX-License-Identifier: GPL-3.0-or-later
"""Local ComfyUI boundary: unload barriers and non-pickle geometry/material interchange."""

import gc
import json
import os
import re
from pathlib import Path

import comfy.model_management as mm
import folder_paths
import numpy as np
import torch
from comfy_extras.nodes_mesh_postprocess import get_mesh_batch_item


class EffigyUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"value": ("*",)},
            "optional": {"after": ("*",), "start": ("BOOLEAN", {"default": False})},
        }

    RETURN_TYPES = ("*",)
    FUNCTION = "execute"
    CATEGORY = "Effigy"

    def execute(self, value, after=None, start=False):
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()
        if mm.current_loaded_models:
            raise RuntimeError("A previous generation stage still has loaded models")
        if (
            start and torch.cuda.is_available()
        ):  # a run's first barrier: its memory cap, its own peak
            if budget := os.environ.get("EFFIGY_COMFYUI_VRAM_GB"):
                total = torch.cuda.get_device_properties(0).total_memory
                torch.cuda.set_per_process_memory_fraction(
                    min(0.95, float(budget) * 2**30 / total), 0
                )
            torch.cuda.reset_peak_memory_stats()
        return (value,)


class EffigyExport3D:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw_mesh": ("MESH",),
                "repaired_mesh": ("MESH",),
                "mesh": ("MESH",),
                "voxel_colors": ("VOXEL",),
                "fov": ("FLOAT",),
                "token": ("STRING",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "Effigy"

    def execute(self, raw_mesh, repaired_mesh, mesh, voxel_colors, fov, token):
        if not re.fullmatch(r"[a-f0-9]{32}", token):
            raise ValueError("Invalid generation token")
        arrays = {}
        for prefix, item in [("raw", raw_mesh), ("repaired", repaired_mesh), ("mesh", mesh)]:
            v, f, *_ = get_mesh_batch_item(item, 0)
            arrays[prefix + "_vertices"] = v.detach().float().cpu().numpy()
            arrays[prefix + "_faces"] = f.detach().int().cpu().numpy()
        arrays.update(
            coords=voxel_colors.data[:, -3:].detach().int().cpu().numpy(),
            attrs=voxel_colors.voxel_colors.detach().float().cpu().numpy(),
            resolution=np.int32(voxel_colors.resolution),
            fov=np.float32(fov),
            peak_vram_gb=np.float32(
                torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
            ),
        )
        filename = "effigy_" + token + ".npz"
        np.savez(Path(folder_paths.get_temp_directory()) / filename, **arrays)
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()
        return {"ui": {"files": [{"filename": filename, "subfolder": "", "type": "temp"}]}}


class EffigyMultiViewConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"clip_vision_model": ("CLIP_VISION",), "cameras": ("STRING",)},
            "optional": {name: ("IMAGE",) for name in ("front", "left", "back", "right")},
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    FUNCTION = "execute"
    CATEGORY = "Effigy"

    def execute(self, clip_vision_model, cameras, **views):
        from comfy.utils import common_upscale
        from comfy_extras.nodes_trellis2 import _build_pixal3d_conditioning

        rig = json.loads(cameras)
        frames = rig["frames"]
        items = []
        for frame in frames:
            image = views[frame["view"]][..., :3]
            image = common_upscale(image.movedim(-1, 1), 1024, 1024, "lanczos", "disabled").movedim(
                1, -1
            )
            items.append(image)
        c2w = torch.tensor([f["transform_matrix"] for f in frames], dtype=torch.float32)
        result = _build_pixal3d_conditioning(
            clip_vision_model,
            torch.cat(items),
            c2w,
            torch.full((len(frames),), float(rig["camera_angle_x"])),
            torch.ones(1),
            num_views=len(frames),
        )
        return tuple(result.result)


NODE_CLASS_MAPPINGS = {
    "EffigyUnload": EffigyUnload,
    "EffigyExport3D": EffigyExport3D,
    "EffigyMultiViewConditioning": EffigyMultiViewConditioning,
}
