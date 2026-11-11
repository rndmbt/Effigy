"""Image → 3D (IMPLEMENTATION.md M2) and retexture (M12) with TRELLIS.2.

op "generate" on a `prep` version → hi.glb, ovoxel.npz, hi_trellis.ply, preview.png. params: {weights: fast
(TRELLIS.2-4B) | faithful (Pixal3D), pipeline_type, steps, seed, texture_size, faces, height_m, fov (faithful:
camera field of view, radians), background: auto | black | grey | white}. backend "comfyui" (steps/_comfy3d.py)
estimates the camera unless camera_auto is false. The models see the cut-out on a flat
background, black in their own preprocessing: a black cat's back then reads as background and comes out as
a hole. Measured on a black-and-white cat, same seed: Fast on black lost the whole black body (1.0 m² of
surface), on grey it was whole (2.3 m²); Faithful (it projects its voxels into the picture) was whole on
both. `auto`: grey for both generators; explicit black and white remain available.
op "texture" on any version with a UV-mapped .glb → the same mesh and UVs with a new PBR set. params: {image:
a prep version of this asset (default: the one up the chain), resolution: 1024|512, steps, seed, texture_size}.
One-shot or resident (`--serve`): the loaded pipeline stays in PIPES between requests. The GPU-only pieces are
the `_load*` functions and `_simplify_unwrap`; everything else also runs on the CPU (tests).
"""

import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image

from steps._blender import BlenderError, run_blender
from steps._common import (
    PIXAL3D_TO_GLTF,
    TRELLIS_TO_GLTF,
    StepError,
    main,
    progress,
    save_ovoxel,
    trellis_to_library,
)
from steps._image import BACKGROUNDS, dark_share, on_background
from steps._mesh import drop_small_parts, face_outward, fill_holes

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("EFFIGY_MODELS", ROOT / "models"))
DINOV3 = "facebook/dinov3-vitl16-pretrain-lvd1689m"
WEIGHTS = {
    "fast": {
        "pipeline_types": {"512", "1024", "1024_cascade", "1536_cascade"},
        "to_gltf": TRELLIS_TO_GLTF,
        "licenses": ["microsoft/TRELLIS.2-4B", "microsoft/TRELLIS-image-large", DINOV3],
        "model": {
            "id": "microsoft/TRELLIS.2-4B",
            "revision": "af44b45f2e35a493886929c6d786e563ec68364d",
        },
    },
    "faithful": {
        "pipeline_types": {"1024_cascade", "1536_cascade"},  # Pixal3D only has the cascades
        "to_gltf": PIXAL3D_TO_GLTF,
        "licenses": ["TencentARC/Pixal3D", DINOV3, "valeoai/NAF"],
        "model": {
            "id": "TencentARC/Pixal3D",
            "revision": "b0cb2e1b794cab9aa0ac38a95d794a4d9337437f",
        },
    },
}
# Pixal3D's inference.py builds one proj image extractor per stage (they are not in its pipeline.json)
PIXAL3D_COND = {
    "ss": {"image_size": 512, "grid_resolution": 16},
    "shape_512": {
        "image_size": 512,
        "grid_resolution": 32,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "shape_1024": {
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 512,
    },
    "tex_1024": {
        "image_size": 1024,
        "grid_resolution": 64,
        "use_naf_upsample": True,
        "naf_target_size": 1024,
    },
}
PIPES: dict = {}  # weights → loaded pipeline (one at a time: VRAM)


def _local(folder: str, group: str) -> Path:
    root = MODELS / folder
    if not (root / "pipeline.json").is_file():
        raise StepError(
            "missing_input",
            f"{root} missing: run `uv run --group ml python -m tools.download_models {group}`",
        )
    return root


def _load(weights: str):
    """The pinned local copies tools/download_models.py made (no network here)."""
    if weights in ("faithful", "faithful_mv"):
        return _load_pixal3d(weights == "faithful_mv")
    if weights == "texture":
        return _load_texturing()
    from engine.trellis2.patches import prepare

    prepare(MODELS)
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    root = _local("TRELLIS.2-4B", "fast")
    cfg = json.loads((root / "pipeline.json").read_text())
    args = cfg["args"]
    # the sparse-structure decoder lives in TRELLIS-image-large; DINOv3 is gated: both downloaded locally
    args["models"]["sparse_structure_decoder"] = str(
        MODELS / "TRELLIS-image-large" / "ckpts" / "ss_dec_conv3d_16l8_fp16"
    )
    args["image_cond_model"]["args"]["model_name"] = str(MODELS / "dinov3-vitl16")
    # only pipeline_type "512" (the OOM fallback) uses it: ~2.6 GB less RAM; loaded on demand in run()
    args["models"].pop(TEX_512, None)
    (root / "effigy_pipeline.json").write_text(json.dumps(cfg, indent=2))
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(str(root), "effigy_pipeline.json")
    pipe.cuda()  # low_vram (default): each model moves to the GPU only for its stage
    return pipe


def _load_pixal3d(mv: bool = False):
    """mv: the four-view checkpoints (`download_models multiview`), same cascade, views averaged."""
    import torch

    from engine.pixal3d.patches import prepare

    prepare(MODELS)
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline, Pixal3DMVImageTo3DPipeline
    from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj as proj

    root = _local("Pixal3D", "faithful")
    if mv:
        if not (root / "pipeline_mv.json").is_file():
            raise StepError(
                "missing_input",
                "Pixal3D multi-view weights missing: python -m tools.download_models multiview",
            )
        pipe = Pixal3DMVImageTo3DPipeline.from_pretrained(str(root), "pipeline_mv.json")
    else:
        pipe = Pixal3DImageTo3DPipeline.from_pretrained(str(root))
    extractor = proj.DinoV3ProjMultiViewFeatureExtractor if mv else proj.DinoV3ProjFeatureExtractor
    # one shared DINOv3 underneath the four extractors (engine/pixal3d/patches.py)
    for stage, cfg in PIXAL3D_COND.items():
        ext = extractor(str(MODELS / "dinov3-vitl16"), **cfg).eval()
        setattr(pipe, f"image_cond_model_{stage}", ext)
    pipe._device, pipe.low_vram = torch.device("cuda"), True  # each stage moves on and off the GPU
    return pipe


def _load_texturing():
    """Upstream's shape-conditioned texturing models, from the same TRELLIS.2-4B download. We use its stages
    (encode shape, sample, decode), not its run(): that one re-unwraps the UVs (PR #173) and bakes with
    nvdiffrast."""
    from engine.trellis2.patches import prepare

    prepare(MODELS)
    from trellis2.pipelines import Trellis2TexturingPipeline

    root = _local("TRELLIS.2-4B", "fast")
    cfg = json.loads((root / "texturing_pipeline.json").read_text())
    cfg["args"]["image_cond_model"]["args"]["model_name"] = str(MODELS / "dinov3-vitl16")
    (root / "effigy_texturing_pipeline.json").write_text(json.dumps(cfg, indent=2))
    pipe = Trellis2TexturingPipeline.from_pretrained(str(root), "effigy_texturing_pipeline.json")
    pipe.cuda()  # low_vram: each model moves to the GPU only for its stage
    return pipe


TEX_512 = "tex_slat_flow_model_512"


def pipeline(weights: str):
    if weights not in PIPES:
        # one pipeline at a time: a faithful -> fast switch without this left 15.8 of 16 GB in use and the
        # next generate crawled in shared memory; the collect also frees what a failed run's traceback kept
        # alive after `run` evicted it
        import gc

        import torch

        PIPES.clear()
        gc.collect()
        torch.cuda.empty_cache()
        PIPES[weights] = _load(weights)
    return PIPES[weights]


def _simplify_unwrap(vertices, faces, target_faces: int):
    """CuMesh, after upstream to_glb's non-remesh branch: fill holes, simplify ×3 then to target with clean-ups,
    unwrap. Returns (low verts, faces, uvs, normals, project) where project maps points onto the original
    surface (BVH closest point), so texels sample the volume where the detailed mesh is."""
    import cumesh

    m = cumesh.CuMesh()
    m.init(vertices.cuda().float(), faces.cuda().int())
    fill_holes(m, 3e-2)
    hv, hf = m.read()
    bvh = cumesh.cuBVH(hv, hf)
    for target in (target_faces * 3, target_faces):
        m.simplify(target)
        m.remove_duplicate_faces()
        # no repair_non_manifold_edges here (it shatters noisy surfaces; clean makes the mesh manifold) and
        # our own small-part filter instead of CuMesh's, which deleted most of a Pixal3D stump
        drop_small_parts(m, 1e-5)
        fill_holes(m, 3e-2)
    m.unify_face_orientations()
    face_outward(m)
    lv, lf, uv, vmap = m.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": float(np.radians(90.0)),
            "refine_iterations": 0,
            "global_iterations": 1,
            "smooth_strength": 1,
        },
        return_vmaps=True,
    )
    m.compute_vertex_normals()
    normals = m.read_vertex_normals()[vmap.cuda()]

    def project(points):
        _, face, w = bvh.unsigned_distance(points, return_uvw=True)
        return (hv[hf[face.long()]] * w[..., None]).sum(1)

    return lv.cuda(), lf.cuda().long(), uv.cuda(), normals, project, (hv, hf)


def _params(params: dict) -> dict:
    p = (
        {
            "weights": "fast",
            "backend": "native",
            "pipeline_type": "1024_cascade",
            "steps": 12,
            "seed": random.randrange(2**31),
            "texture_size": 2048,
            "faces": 200_000,
            "height_m": 1.0,
            "fov": 0.2,  # Pixal3D's suggestion for product-style shots
            "camera_auto": True,  # backend comfyui: MoGe-2 estimates the camera; false uses `fov`
            "multiview": True,  # faithful + a prep "views" version (cameras.json): all views, else the front
            "background": "auto",  # behind the cut-out: auto (grey) | black | grey | white
        }
        | params
    )
    ok = (
        p["weights"] in WEIGHTS
        and p["backend"] in ("native", "comfyui")
        and p["pipeline_type"] in WEIGHTS[p["weights"]]["pipeline_types"]
        and 1 <= int(p["steps"]) <= 50
        and p["texture_size"] in (256, 512, 1024, 2048, 4096)
        and 1000 <= int(p["faces"]) <= 2_000_000
        and float(p["height_m"]) > 0
        and 0.05 <= float(p["fov"]) <= 1.6
        and p["background"] in ("auto", *BACKGROUNDS)
    )
    if not ok:
        raise StepError("bad_params", f"bad trellis2 params: {params}")
    return p


def run(op, in_dir, out_dir, params):
    """A failure inside the models evicts the resident pipeline: `low_vram` moves a stage back to the CPU
    only after it succeeds, so an OOM mid-sample left the flow model on the card for every later job
    (code review 2026-10-05). The next request reloads (~30 s) into a clean card instead. A StepError
    (bad input, before any model ran) keeps the pipeline."""
    try:
        return _run(op, in_dir, out_dir, params)
    except StepError:
        raise
    except Exception:
        if PIPES:
            import gc

            import torch

            PIPES.clear()
            gc.collect()
            torch.cuda.empty_cache()
        raise


def _framed(src: Path, background: str, weights: str = "fast"):
    """The prep's framed cut-out on its background, and {"background", "dark_share"} for the metrics."""
    rgba_file = src / "image.png"
    if not rgba_file.is_file():  # a prep from before M36's option: its black composite as it is
        return Image.open(src / "image_rgb.png").convert("RGB"), {"background": "black"}
    rgba = np.asarray(Image.open(rgba_file).convert("RGBA"))
    dark = dark_share(rgba)
    if background == "auto":
        background = "grey"
    rgb = on_background(rgba, BACKGROUNDS[background])
    return Image.fromarray(rgb, "RGB"), {"background": background, "dark_share": round(dark, 3)}


def _run(op, in_dir, out_dir, params):
    if op == "texture":
        return texture(in_dir, out_dir, params)
    if op not in (
        "generate",
        "run",
    ):  # "run": nothing said, so generate (a texture run names its op)
        raise StepError("bad_params", f"unknown op {op!r} (generate, texture)")
    p = _params(params)
    src = Path(in_dir or "") / "image_rgb.png"
    if not in_dir or not src.is_file():
        raise StepError("missing_input", "generate needs a prep version (image_rgb.png)")
    if p.get("backend") == "comfyui":
        from steps import _comfy3d

        return _comfy3d.run(in_dir, out_dir, p)
    import torch

    from steps._bake import bake_volume, rasterize_uv, write_glb

    t0, out = time.time(), Path(out_dir)
    cams = Path(in_dir) / "cameras.json"
    mv = p["weights"] == "faithful" and bool(p["multiview"]) and cams.is_file()
    image, bg = _framed(Path(in_dir), p["background"], p["weights"])
    progress(0.02, f"loading TRELLIS.2 ({p['weights']}{', multi-view' if mv else ''})")
    pipe = pipeline("faithful_mv" if mv else p["weights"])
    if p["weights"] == "fast" and p["pipeline_type"] == "512" and TEX_512 not in pipe.models:
        from trellis2 import models

        pipe.models[TEX_512] = models.from_pretrained(
            str(_local("TRELLIS.2-4B", "fast") / "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16")
        ).eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    progress(0.1, f"sampling {p['pipeline_type']}, {p['steps']} steps, seed {p['seed']}")
    steps = {"steps": int(p["steps"])}
    extra = {}
    if p["weights"] == "faithful":
        # Pixal3D projects its voxels into the image: the unit cube's half-width fills the frame at this
        # distance (inference.py's distance_from_fov with extend_pixel 0 reduces to 0.5 / tan(fov / 2))
        fov = float(p["fov"])
        extra["camera_params"] = {
            "camera_angle_x": fov,
            "distance": 0.5 / math.tan(fov / 2),
            "mesh_scale": 1.0,
        }
    sampling = {
        "seed": int(p["seed"]),
        "pipeline_type": p["pipeline_type"],
        "sparse_structure_sampler_params": steps,
        "shape_slat_sampler_params": steps,
        "tex_slat_sampler_params": steps,
    }
    if mv:
        mesh = pipe.run_mv(
            _views(json.loads(cams.read_text(encoding="utf-8")), Path(in_dir)), **sampling
        )[0]
    else:  # prep already framed it like TRELLIS's preprocessing, plus the 4 % margin that measured better
        mesh = pipe.run(image, preprocess_image=False, **sampling, **extra)[0]
    t2 = time.time()

    progress(0.75, "saving the volume")
    resolution = round(1 / float(mesh.voxel_size))
    layout = {k: [s.start, s.stop] for k, s in mesh.layout.items()}
    save_ovoxel(
        out / "ovoxel.npz",
        mesh.coords.cpu().numpy(),
        mesh.attrs.float().cpu().numpy(),
        layout,
        resolution,
    )

    progress(0.8, f"simplify to {p['faces']} faces, unwrap, bake {p['texture_size']}²")
    lv, lf, uv, normals, project, (hv, hf) = _simplify_unwrap(
        mesh.vertices, mesh.faces, int(p["faces"])
    )
    import trimesh

    trimesh.Trimesh(hv.cpu().numpy(), hf.cpu().numpy(), process=False).export(
        out / "hi_trellis.ply"
    )
    face_id, bary = rasterize_uv(uv, lf, int(p["texture_size"]))
    rgba, mr, _ = bake_volume(
        face_id, bary, lv, lf, mesh.coords, mesh.attrs, layout, resolution, project
    )
    hv_np = hv.cpu().numpy()
    w = WEIGHTS[p["weights"]]
    frame = trellis_to_library(hv_np.min(0), hv_np.max(0), float(p["height_m"]), w["to_gltf"])
    write_glb(
        out / "hi.glb",
        lv.cpu().numpy(),
        lf.cpu().numpy(),
        uv.cpu().numpy(),
        rgba,
        mr,
        normals.cpu().numpy(),
        transform=frame,
    )

    outputs = {"mesh": "hi.glb", "ovoxel": "ovoxel.npz", "hi_trellis": "hi_trellis.ply"}
    progress(0.95, "preview")
    try:  # Cycles on the CPU; a generation is never thrown away for a thumbnail
        run_blender(
            "render_thumb",
            [out / "hi.glb"],
            out / "_preview",
            {"size": 512, "samples": 16},
            timeout=300,
        )
        shutil.move(out / "_preview" / "thumb.png", out / "preview.png")
        outputs["preview"] = "preview.png"
    except BlenderError as e:
        print(f"preview skipped: {e}")
    shutil.rmtree(out / "_preview", ignore_errors=True)

    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    metrics = {
        "seconds": round(time.time() - t0, 1),
        "seconds_load": round(t1 - t0, 1),
        "seconds_sample": round(t2 - t1, 1),
        "peak_vram_gb": round(peak, 2),
        "faces_hi": len(hf),
        "faces": len(lf),
        "voxels": len(mesh.coords),
        "seed": int(p["seed"]),
        "weights": p["weights"],
        **bg,  # what the model saw behind the cut-out
        "frame": {
            "library": "gltf_yup_m",
            "to_library_from_source": np.asarray(frame).round(9).tolist(),
        },
    }
    if p["weights"] == "faithful":
        metrics["fov"] = float(p["fov"])
    if mv:
        metrics["fov"] = json.loads(cams.read_text(encoding="utf-8"))["camera_angle_x"]
        metrics["views"] = [
            f["view"] for f in json.loads(cams.read_text(encoding="utf-8"))["frames"]
        ]
    return {
        "outputs": outputs,
        "metrics": metrics,
        "models": [w["model"]],
        "licenses": w["licenses"],
    }


def _views(cams: dict, folder: Path) -> dict:
    """prep "views" output → the bundle Pixal3DMVImageTo3DPipeline.run_mv takes (upstream inference_mv's
    load_views): RGB on black at every stage's input size, c2w per view, view 0 the main one."""
    import torch

    frames = cams["frames"]
    imgs = [Image.open(folder / f["file_path"]).convert("RGB") for f in frames]
    sizes = {cfg["image_size"] for cfg in PIXAL3D_COND.values()}
    n = len(frames)
    return {
        "images": {
            size: torch.stack(
                [
                    torch.from_numpy(
                        np.asarray(i.resize((size, size), Image.LANCZOS), np.float32) / 255
                    ).permute(2, 0, 1)
                    for i in imgs
                ]
            )[None]
            for size in sizes
        },
        "camera_angle_x": torch.full((1, n), float(cams["camera_angle_x"])),
        "camera_distance": torch.full((1, n), float(cams["camera_distance"])),
        "transform_matrix": torch.tensor(
            [f["transform_matrix"] for f in frames], dtype=torch.float32
        )[None],
        "mesh_scale": 1.0,
    }


def _image_version(src: Path, name: str | None) -> Path:
    """The prep version to texture from: `name` in this asset, or the nearest one up the chain (the image
    the mesh was generated from)."""
    if name:
        if not re.fullmatch(r"\d{3}_[a-z0-9_]+", name) or not (src.parent / name).is_dir():
            raise StepError("bad_params", f"no version {name!r} in this asset")
        d = src.parent / name
        if not (d / "image_rgb.png").is_file():
            raise StepError(
                "missing_input", f"{name} has no image_rgb.png: pick or upload one, then prep"
            )
        return d
    d = src
    while not (d / "image_rgb.png").is_file():
        parent = json.loads((d / "meta.json").read_text(encoding="utf-8")).get("parent")
        if not parent:
            raise StepError(
                "missing_input", "no prep image up the chain: pass params.image = a prep version"
            )
        d = src.parent / parent
    return d


def texture(in_dir, out_dir, params):
    """New PBR set for an existing UV-mapped mesh, conditioned on its shape and an image. Vertices, faces and
    UVs are written back unchanged: the textures are baked onto them by steps/_bake.py, as in M3."""
    p = {
        "image": None,
        "resolution": 1024,
        "steps": 12,
        "seed": random.randrange(2**31),
        "texture_size": 2048,
    } | params
    if not (
        p["resolution"] in (512, 1024)
        and 1 <= int(p["steps"]) <= 50
        and p["texture_size"] in (256, 512, 1024, 2048, 4096)
    ):
        raise StepError("bad_params", f"bad texture params: {params}")
    if not in_dir:
        raise StepError("missing_input", "texture needs a version with a UV-mapped mesh")
    import torch
    import trimesh

    from steps._bake import bake_volume, rasterize_uv, write_glb

    src, out, t0 = Path(in_dir), Path(out_dir), time.time()
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    glb = next((f for f in meta["outputs"].values() if f.endswith(".glb")), None)
    if not glb:
        raise StepError("missing_input", f"{meta['version']} has no .glb")
    scene = trimesh.load(src / glb, process=False)
    if (
        len(getattr(scene, "geometry", {None: scene})) != 1
    ):  # several would be merged and their UVs repacked
        raise StepError("missing_input", f"{glb}: join it into one mesh with one material first")
    mesh = trimesh.load(src / glb, force="mesh", process=False)
    if getattr(mesh.visual, "uv", None) is None:
        raise StepError("missing_input", f"{glb} has no UVs: run the uv step first")
    img_dir = _image_version(src, p["image"])
    image = Image.open(img_dir / "image_rgb.png").convert("RGB")

    # library frame → the texturing model's: centred, longest side 0.99999, Z up (upstream preprocess_mesh)
    v = np.asarray(mesh.vertices, float)
    lo, hi = v.min(0), v.max(0)
    move, scale = np.eye(4), np.diag([0.99999 / (hi - lo).max()] * 3 + [1.0])
    move[:3, 3] = -(lo + hi) / 2
    to_trellis = TRELLIS_TO_GLTF.T @ scale @ move
    v_t = v @ to_trellis[:3, :3].T + to_trellis[:3, 3]

    res = int(p["resolution"])
    progress(0.05, "loading the TRELLIS.2 texturing models")
    pipe = pipeline("texture")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    progress(0.15, f"sampling textures at {res}, {p['steps']} steps, seed {p['seed']}")
    torch.manual_seed(int(p["seed"]))
    # upstream's run() is no_grad; its stages called one by one aren't: encode_shape_slat kept the shape
    # encoder's autograd graph (25.9 GB on a 16 GB card -> 34 s/step, then OOM). Off: 3.8 GB peak.
    with torch.inference_mode():
        cond = pipe.get_cond([image], res)
        shape_slat = pipe.encode_shape_slat(trimesh.Trimesh(v_t, mesh.faces, process=False), res)
        flow = pipe.models[f"tex_slat_flow_model_{res}"]
        vox = pipe.decode_tex_slat(
            pipe.sample_tex_slat(cond, flow, shape_slat, {"steps": int(p["steps"])})
        )
    coords, attrs = vox.coords[:, 1:], vox.feats.float()
    layout = {k: [s.start, s.stop] for k, s in pipe.pbr_attr_layout.items()}

    progress(0.8, f"baking {p['texture_size']}² onto the mesh's own UVs")
    save_ovoxel(out / "ovoxel.npz", coords.cpu().numpy(), attrs.cpu().numpy(), layout, res)
    trimesh.Trimesh(v_t, mesh.faces, process=False).export(
        out / "hi_trellis.ply"
    )  # M3 re-bake surface
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    uv = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=dev)
    uv[:, 1] = 1 - uv[:, 1]  # trimesh's OpenGL UVs → glTF UVs (what the file holds)
    f = torch.tensor(mesh.faces, dtype=torch.long, device=dev)
    face_id, bary = rasterize_uv(uv, f, int(p["texture_size"]))
    v_dev = torch.tensor(v_t, dtype=torch.float32, device=dev)
    rgba, mr, m = bake_volume(face_id, bary, v_dev, f, coords.to(dev), attrs.to(dev), layout, res)
    orm = mr.copy()
    orm[..., 0] = 255  # R = AO (none), G = roughness, B = metallic
    Image.fromarray(rgba, "RGBA").save(out / "basecolor.png")
    Image.fromarray(orm, "RGB").save(out / "orm.png")
    write_glb(out / "mesh.glb", v, mesh.faces, uv.cpu().numpy(), rgba, orm, mesh.vertex_normals)

    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    image_meta = json.loads((img_dir / "meta.json").read_text(encoding="utf-8"))
    metrics = {
        "seconds": round(time.time() - t0, 1),
        "peak_vram_gb": round(peak, 2),
        "resolution": res,
        "texture_size": int(p["texture_size"]),
        "coverage": round(float(m.mean()), 3),
        "seed": int(p["seed"]),
        "image": img_dir.name,
        "frame": {
            "library": "gltf_yup_m",
            "to_library_from_source": np.linalg.inv(to_trellis).round(9).tolist(),
        },
    }
    outputs = {
        "mesh": "mesh.glb",
        "basecolor": "basecolor.png",
        "orm": "orm.png",
        "ovoxel": "ovoxel.npz",
        "hi_trellis": "hi_trellis.ply",
    }
    licenses = sorted({"microsoft/TRELLIS.2-4B", DINOV3, *image_meta.get("licenses", [])})
    return {
        "outputs": outputs,
        "metrics": metrics,
        "models": [WEIGHTS["fast"]["model"]],
        "licenses": licenses,
    }


if __name__ == "__main__":
    main(run)
