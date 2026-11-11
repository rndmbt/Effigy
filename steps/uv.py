"""UV-unwrap: UVgami OptCuts by default, or CuMesh charts + xatlas packing → mesh.glb with UVs.

params: {cone_deg: 90, refine: 0, global_iterations: 1, smooth: 1, unwrapper: "uvgami" (default)|"cumesh",
priority: "BALANCED"|"LESS_STRETCH"|"FEWER_SEAMS", symmetry: true, timeout: 1800}
unwrapper "uvgami" (2026-11-04): UVgami's OptCuts unwrap in Blender (blender_ops/uvgami.py), installed once by
server/uvgami_install (EFFIGY_UVGAMI), on the parent's quads.obj when it has one (the quads kept). On a generated
model's quad mesh: 1 island and 1.6 % of the edge length in seams against CuMesh+xatlas's 95 islands and 9.2 %,
texel size more even (0.20 against 0.29 log2), a little more stretch (1.21 against 1.07).
"""

import json
from pathlib import Path

import numpy as np

from steps._common import StepError, main, progress
from steps._mesh import read_glb, uv_report, weld, write_glb


def _cumesh_unwrap(v, f, p: dict):
    import cumesh
    import torch

    m = cumesh.CuMesh()
    m.init(torch.tensor(v, dtype=torch.float32).cuda(), torch.tensor(f, dtype=torch.int32).cuda())
    v2, f2, uv, _ = m.uv_unwrap(
        compute_charts_kwargs={
            "threshold_cone_half_angle_rad": float(np.radians(p.get("cone_deg", 90))),
            "refine_iterations": int(p.get("refine", 0)),
            "global_iterations": int(p.get("global_iterations", 1)),
            "smooth_strength": float(p.get("smooth", 1)),
        },
        return_vmaps=True,
    )
    return v2.cpu().numpy(), f2.cpu().numpy(), uv.cpu().numpy()


def _uvgami(src: Path, out: Path, p: dict):
    """(vertices, faces, glTF UVs) of `src` unwrapped by UVgami in Effigy's Blender, its own profile and engine."""
    import os
    import shutil

    from steps._blender import BlenderError, run_blender

    home = Path(os.environ.get("EFFIGY_UVGAMI") or Path(__file__).resolve().parents[1] / "uvgami")
    profile, engine = home / "profile", home / "engine" / "bin"
    if not (profile / "extensions" / "user_default" / "uvgami").is_dir() or not engine.is_dir():
        raise StepError(
            "missing_input", "UVgami not installed: run `uv run python -m server.uvgami_install`"
        )
    os.environ.update(BLENDER_USER_RESOURCES=str(profile), UVGAMI_ENGINE_DIR=str(engine))
    progress(0.2, "UVgami (OptCuts) unwrapping")
    keys = {k: p[k] for k in ("priority", "symmetry", "timeout", "proxy_faces") if k in p}
    try:
        run_blender(
            "uvgami", [src], out / "_uvg", keys, timeout=float(p.get("timeout", 1800)) + 120
        )
    except BlenderError as e:
        raise StepError("failed", str(e)) from e
    v, f, uv = read_glb(out / "_uvg" / "mesh.glb")
    shutil.rmtree(out / "_uvg")
    if uv is None:
        raise StepError("failed", "UVgami returned no UVs")
    return v, f, uv


def run(op, in_dir, out_dir, params):
    if not in_dir:
        raise StepError("missing_input", "uv needs a parent version with a mesh")
    meta = json.loads((Path(in_dir) / "meta.json").read_text(encoding="utf-8"))
    glb = next((f for f in meta["outputs"].values() if f.endswith(".glb")), None)
    if not glb:
        raise StepError("missing_input", f"{meta['version']} has no .glb")
    unwrapper = params.get("unwrapper", "uvgami")
    if unwrapper not in ("cumesh", "uvgami"):
        raise StepError("bad_params", "unwrapper is cumesh or uvgami")
    if unwrapper == "uvgami":
        quads = meta["outputs"].get("quads")
        v, f, uv = _uvgami(Path(in_dir) / (quads or glb), Path(out_dir), params)
    else:
        v, f = weld(*read_glb(Path(in_dir) / glb)[:2])
        progress(0.2, f"unwrapping {len(f)} faces")
        v, f, uv = _cumesh_unwrap(v, f, params)
    write_glb(Path(out_dir) / "mesh.glb", v, f, uv)
    rep = uv_report(uv, f, v)
    (Path(out_dir) / "uv_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    if not rep["in_0_1"]:
        raise StepError("failed", "unwrap produced UVs outside [0, 1]")
    return {
        "outputs": {"mesh": "mesh.glb", "uv_report": "uv_report.json"},
        "metrics": {"islands": rep["islands"], "unwrapper": unwrapper},
    }


if __name__ == "__main__":
    main(run)
