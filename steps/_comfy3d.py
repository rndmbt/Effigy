"""Generate through native ComfyUI nodes; keep UVgami and the established library/bake contracts."""

import json
import math
import shutil
import time
import urllib.parse
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from steps import _comfy
from steps._common import StepError, progress, save_ovoxel, trellis_to_library

ROOT = Path(__file__).resolve().parents[1]
LAYOUT = {"base_color": [0, 3], "metallic": [3, 4], "roughness": [4, 5], "alpha": [5, 6]}


def graph(src: Path, p: dict) -> tuple[dict, dict]:
    """The shipped workflow bound to this job, without the stages it does not use, and the framing record."""
    from steps.trellis2 import _framed

    g = json.loads((ROOT / "workflows/pixal3d_generate.json").read_text(encoding="utf-8"))
    image, info = _framed(src, p["background"], p["weights"])
    g["122"]["inputs"]["image"] = _comfy._upload(image)
    g["output"]["inputs"]["token"] = uuid.uuid4().hex
    for sampler in ("3", "18", "23", "12"):  # structure, shape, upsampled shape, materials
        g[sampler]["inputs"].update(seed=int(p["seed"]), steps=int(p["steps"]))
    g["241"]["inputs"].update(
        {
            "resolution": int(p.get("repair_resolution", 768)),
            "fill": float(p.get("repair_fill", 20)),
            "smooth_iters": int(p.get("repair_smooth", 2)),
        }
    )
    resolution, _, cascade = p["pipeline_type"].partition("_")
    if cascade:
        g["94"]["inputs"]["target_resolution"] = int(resolution)
    else:  # one shape stage: materials and decode read it directly, the upsample stage drops out below
        g["119"]["inputs"]["resolution"] = "64" if resolution == "1024" else "32"
        g["92"]["inputs"]["samples"] = ["unload_shape_sample", 0]
        g["98"]["inputs"].update(positive=["91", 0], negative=["91", 1])
        g["unload_shape_decode"]["inputs"]["value"] = ["18", 0]
    # the camera: none for Fast, the prep's own rig for a turnaround, else estimated (MoGe) or the given fov
    if p["weights"] == "fast":
        g["319"]["inputs"]["unet_name"] = "trellis_2_int8_convrot.safetensors"
        g["298"] = {
            "class_type": "Trellis2Conditioning",
            "inputs": {"clip_vision_model": ["15", 0], "image": ["begin", 0]},
        }
        g["output"]["inputs"]["fov"] = 0.0
    elif (src / "cameras.json").is_file() and p["multiview"]:
        cams = json.loads((src / "cameras.json").read_text(encoding="utf-8"))
        views = {"clip_vision_model": ["15", 0], "cameras": json.dumps(cams)}
        for k, frame in enumerate(cams["frames"]):
            name = frame["view"]
            if name not in ("front", "left", "back", "right"):
                raise StepError(
                    "bad_params", f"ComfyUI Pixal3D supports front/left/back/right, not {name!r}"
                )
            picture = Image.open(src / frame["file_path"]).convert("RGB")
            g["view_" + name] = {
                "class_type": "LoadImage",
                "inputs": {"image": _comfy._upload(picture)},
            }
            views[name] = ["view_" + name, 0]
            if k == 0:  # the first barrier sits behind one of the pictures the run reads
                g["begin"]["inputs"]["value"], views[name] = views[name], ["begin", 0]
        g["298"] = {"class_type": "EffigyMultiViewConditioning", "inputs": views}
        g["319"]["inputs"]["unet_name"] = "pixal3d_multiview_int8_convrot.safetensors"
        g["output"]["inputs"]["fov"] = math.degrees(cams["camera_angle_x"])
        info["views"] = [f["view"] for f in cams["frames"]]
    elif not p["camera_auto"]:
        g["298"]["inputs"]["camera_angle_x"] = g["output"]["inputs"]["fov"] = math.degrees(
            float(p["fov"])
        )
    needed = set()

    def visit(node):
        if node not in needed:
            needed.add(node)
            for value in g[node]["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    visit(value[0])

    visit("output")
    return {k: n for k, n in g.items() if k in needed}, info


def sample(src: Path, p: dict) -> tuple[dict, dict]:
    """Run the graph; its arrays (fetched as a plain .npz, never a pickle or a path it names) and the
    framing record."""
    g, info = graph(src, p)
    progress(0.1, "ComfyUI: camera, shape, materials and high-poly reconstruction")
    with _comfy.spill_watch() as added:
        outputs = _comfy.execute(g)
    info["shared_gb"] = added()
    if (info["shared_gb"] or 0) > _comfy.SPILL_GB:
        raise StepError(
            "oom",
            f"Generation added {info['shared_gb']:.2f} GB of shared video memory; "
            "lower the resolution or close other GPU applications",
        )
    expected = {
        "filename": f"effigy_{g['output']['inputs']['token']}.npz",
        "subfolder": "",
        "type": "temp",
    }
    if outputs.get("output", {}).get("files") != [expected]:
        raise StepError("failed", "ComfyUI did not return this generation's geometry and materials")
    data = _comfy._call("/view?" + urllib.parse.urlencode(expected), raw=True)
    _comfy.release()
    with np.load(BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, info


def _version(folder: Path, step: str, parent: str | None, outputs: dict) -> None:
    """A version record the bake step can walk, for the temporary chain under the output folder."""
    meta = {"version": folder.name, "step": step, "parent": parent, "outputs": outputs}
    if parent is None:  # the generation itself: baked in its own unit frame
        meta["frame"] = {"to_library_from_source": np.eye(4).tolist()}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def run(in_dir, out_dir, p: dict) -> dict:
    """Generate in ComfyUI, then finish with the existing steps: UVgami on the high-poly mesh, the bake of the
    generated PBR volume and of the repaired surface's normals."""
    import trimesh

    from steps import bake
    from steps._bake import write_glb as write_textured
    from steps._blender import BlenderError, run_blender
    from steps._mesh import (
        fit_to_surface,
        mesh_report,
        smooth_normals,
        split_creases,
        uv_report,
        write_glb,
    )
    from steps.remesh import _cumesh_simplify
    from steps.trellis2 import WEIGHTS
    from steps.uv import _uvgami

    started, out = time.time(), Path(out_dir)
    arrays, info = sample(Path(in_dir), p)
    raw_v, raw_f = arrays["raw_vertices"], arrays["raw_faces"]
    hi_v, hi_f = arrays["repaired_vertices"], arrays["repaired_faces"]
    # the rebuild back onto the generator's surface (no voxel steps), its cell as comfy_nodes' EffigyRemesh's
    progress(0.55, "fitting the watertight rebuild back onto the generated surface")
    res = int(p.get("repair_resolution", 768))
    cell = (res + 3) / res * 1.02 * float(np.ptp(raw_v, 0).max()) / res
    repaired = mesh_report(hi_v, hi_f)
    hi_v, fit = fit_to_surface(hi_v, hi_f, raw_v, raw_f, cell)
    mesh_v, mesh_f = _cumesh_simplify(hi_v, hi_f, int(p["faces"]), True)
    frame = trellis_to_library(hi_v.min(0), hi_v.max(0), float(p["height_m"]), np.eye(4))
    save_ovoxel(
        out / "ovoxel.npz", arrays["coords"], arrays["attrs"], LAYOUT, int(arrays["resolution"])
    )
    trimesh.Trimesh(raw_v, raw_f, process=False).export(out / "hi_trellis.ply")
    trimesh.Trimesh(hi_v @ frame[:3, :3].T + frame[:3, 3], hi_f, process=False).export(
        out / "hi_clean.ply"
    )
    geometry = {
        "raw": mesh_report(raw_v, raw_f),
        "repaired": repaired,
        "fitted": mesh_report(hi_v, hi_f) | fit,
        "final": mesh_report(mesh_v, mesh_f),
    }
    (out / "geometry_report.json").write_text(json.dumps(geometry, indent=2), encoding="utf-8")

    # the real uv and bake steps on a temporary chain, all of it in the generation's unit frame
    work = out / "_finish"
    source, uv_dir, baked = (work / n for n in ("000_source", "001_uv", "002_bake"))
    for d in (source, uv_dir, baked):
        d.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy(out / "ovoxel.npz", source)
        shutil.copy(out / "hi_trellis.ply", source)
        trimesh.Trimesh(hi_v, hi_f, process=False).export(source / "hi_clean.ply")
        write_glb(source / "mesh.glb", mesh_v, mesh_f)
        _version(
            source,
            "trellis2",
            None,
            {"mesh": "mesh.glb", "ovoxel": "ovoxel.npz", "hi_clean": "hi_clean.ply"},
        )
        progress(0.65, "UVgami: unwrap the high-poly generation")
        v, f, uv = _uvgami(
            source / "mesh.glb",
            uv_dir,
            {
                "timeout": p.get("uv_timeout", 1800),
                "proxy_faces": int(p.get("uv_proxy_faces", 20000)),
                "symmetry": False,
            },
        )
        uvs = uv_report(uv, f, v)
        (out / "uv_report.json").write_text(json.dumps(uvs, indent=2), encoding="utf-8")
        write_glb(uv_dir / "mesh.glb", v, f, uv)
        _version(uv_dir, "uv", source.name, {"mesh": "mesh.glb"})
        progress(0.8, "Baking generated PBR and repaired high-poly normal detail")
        result = bake.run(
            "run",
            uv_dir,
            baked,
            {
                "texture_size": p["texture_size"],
                "supersample": 1,
                "hi_denoise": 6,
                "normal_smooth": 0,
                "ao": False,
            },
        )
        maps = [k for k in ("basecolor", "orm", "normal") if k in result["outputs"]]
        for key in maps:
            shutil.copy(baked / result["outputs"][key], out / f"{key}.png")
        v, f, uv = split_creases(v, f, uv)  # as bake split them: the normal map's frame
        write_textured(
            out / "hi.glb",
            v,
            f,
            uv,
            np.asarray(Image.open(out / "basecolor.png")),
            np.asarray(Image.open(out / "orm.png")),
            smooth_normals(v, f),
            np.asarray(Image.open(out / "normal.png")),
            transform=frame,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    outputs = {
        "mesh": "hi.glb",
        "ovoxel": "ovoxel.npz",
        "hi_trellis": "hi_trellis.ply",
        "hi_clean": "hi_clean.ply",
        "geometry_report": "geometry_report.json",
        "uv_report": "uv_report.json",
    } | {key: f"{key}.png" for key in maps}
    try:
        run_blender(
            "render_thumb",
            [out / "hi.glb"],
            out / "_preview",
            {"size": 512, "samples": 16},
            timeout=300,
        )
        shutil.move(out / "_preview/thumb.png", out / "preview.png")
        outputs["preview"] = "preview.png"
    except BlenderError as e:
        print(f"preview skipped: {e}")
    finally:
        shutil.rmtree(out / "_preview", ignore_errors=True)

    fast, views = p["weights"] == "fast", "views" in info
    estimated = not fast and not views and p["camera_auto"]
    models = [  # the converted checkpoints this run loaded (tools/download_models.py pins the same revisions)
        {"id": "Comfy-Org/Pixal3D", "revision": "f37641be376725d324c56626007cc477f1249a69"},
        *(
            [{"id": "Comfy-Org/TRELLIS.2", "revision": "430a9d09b2416687018c8fe8edced2ad4858a439"}]
            if fast
            else []
        ),
        *(
            [{"id": "Comfy-Org/MoGe", "revision": "1484985258ba7ef45c314144fcf0e90924ffbf44"}]
            if estimated
            else []
        ),
    ]
    return {
        "outputs": outputs,
        "models": models,
        "licenses": WEIGHTS[p["weights"]]["licenses"] + [m["id"] for m in models],
        "metrics": {
            "backend": "comfyui",
            "workflow": "pixal3d_generate",
            "unwrapper": "uvgami",
            "seconds": round(time.time() - started, 1),
            "peak_vram_gb": float(arrays["peak_vram_gb"]),
            "faces": len(f),
            "faces_hi": len(hi_f),
            "seed": int(p["seed"]),
            "weights": p["weights"],
            "fov": math.radians(float(arrays["fov"])),
            "camera_auto": estimated,
            "geometry": geometry,
            "islands": uvs["islands"],
            "frame": {"library": "gltf_yup_m", "to_library_from_source": frame.round(9).tolist()},
            "bake": result["metrics"],
            **info,
        },
    }
