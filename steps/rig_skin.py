"""Skin weights for a fitted skeleton (IMPLEMENTATION.md M20): mesh + skeleton.json → rigged.glb + qa.json.

Input: a rig_fit version (mesh.glb + skeleton.json). Runs in `.venv-skin` (server/jobs.py).
params: {"engine": "auto"|"blender"|"skintokens", "voxels": 120, "rigid": {"Helmet": "Head", "Shield": ""}}
- blender: blender_ops/skin_proxy, bone heat on a closed voxel proxy moved onto the mesh.
- skintokens: steps/_skintokens, the learned model, skin-only on our skeleton (needs the `skin` weights).
- auto (default): both, judged by blender_ops/rig_qa (§3.9: fewest fails, then warnings, then the
  smallest stretch in test poses); the other engine's result stays beside it as rigged_<engine>.glb.
  SkinTokens unavailable (no weights, no GPU) → blender alone, noted in the metrics.
qa.json: {"items": the checklist of the picked engine, "report": its numbers, "engines": every engine's
report, "picked": engine}, items pass|warn|fail like validate's checklist.
A parts version (M32, parts.json from rig_fit) keeps one skinned object per part; a part in `rigid` moves
with one bone ("" = the body bone nearest the part, steps/_rig.nearest_bone): all its weight on that bone,
written into each engine's GLB before the QA judges it.
"""

import json
import shutil
from pathlib import Path

import numpy as np

from steps import _glb, _rig, _templates
from steps._blender import BlenderError, run_blender
from steps._common import StepError, main, progress

# (item, report key, pass if ≤, warn if ≤; above: fail)
LIMITS = [
    ("unweighted", "unweighted_vertices", 0, 0),
    ("weight_sums", "weight_sum_off", 0, 0),
    ("influences", "max_influences", 4, 4),
    ("lr_symmetry", "lr_asymmetry", 0.15, 0.3),
    ("pose_volume", "pose_volume_change_pct_max", 5.0, 15.0),
    ("pose_stretch", "pose_edge_stretch_max", 3.0, 8.0),
]
ENGINES = ("blender", "skintokens")


def checklist(report: dict) -> dict:
    items = {}
    for item, key, ok, warn in LIMITS:
        x = report.get(key)
        status = "warn" if x is None else "pass" if x <= ok else "warn" if x <= warn else "fail"
        items[item] = {"status": status, "value": x}
    return items


def rank(report: dict) -> tuple:
    """Lower is better: fails, warnings, stretch."""
    items = checklist(report)
    n = lambda s: sum(i["status"] == s for i in items.values())
    return n("fail"), n("warn"), report.get("pose_edge_stretch_max") or 0.0


def _creature(src: Path) -> dict:
    """The skinning's and the QA's params for a creature template (M36), {} for the human skeleton."""
    skel = json.loads((src / "skeleton.json").read_text(encoding="utf-8"))
    t = skel.get("template", _templates.HUMAN)
    if t == _templates.HUMAN:
        return {}
    return {
        "torso": _templates.torso(t),
        "limbs": _templates.limbs(t),
        "sides": list(_templates.TEMPLATES[t][1]),
        "yaw": skel.get(
            "yaw", 0
        ),  # rig_qa mirrors the sides across the animal's middle plane by it
    }


def _blender(src: Path, out: Path, voxels: int) -> dict:
    res = run_blender(
        "skin_proxy",
        [src / "mesh.glb", src / "skeleton.json"],
        out / "_blender",
        {"voxels": voxels, "parts": (src / "parts.json").is_file()} | _creature(src),
        timeout=900,
    )
    shutil.move(out / "_blender" / "rigged.glb", out / "rigged_blender.glb")
    shutil.rmtree(out / "_blender", ignore_errors=True)
    return res["report"]


def _skintokens(src: Path, out: Path) -> dict:
    from steps import _skintokens  # the port, importable in .venv-skin only

    return _skintokens.skin(src / "mesh.glb", src / "skeleton.json", out / "rigged_skintokens.glb")


def _qa(out: Path, engine: str, creature: dict) -> dict:
    sym = {k: creature[k] for k in ("sides", "yaw") if k in creature}
    rep = run_blender(
        "rig_qa", [out / f"rigged_{engine}.glb"], out / f"_qa_{engine}", sym, timeout=600
    )["report"]
    shutil.rmtree(out / f"_qa_{engine}", ignore_errors=True)
    return rep


def _loops(src: Path) -> dict | None:
    """Edge loops at the joints (§3.9, _rig.joint_loops) from the source mesh: engine-independent, so it sits
    beside the picked engine's checks and stays out of the ranking. None without a mesh or joints."""
    skel = json.loads((src / "skeleton.json").read_text(encoding="utf-8"))
    if not (src / "mesh.glb").is_file() or not skel.get("joints"):
        return None
    from steps._mesh import read_glb, weld

    try:
        v, _ = weld(*read_glb(src / "mesh.glb")[:2])
        return _rig.joint_loops(v, skel["joints"])
    except (OSError, ValueError, KeyError):  # advisory: the engines' results are in hand, keep them
        return None


def rigid_bones(src: Path, rigid) -> dict[str, str]:
    """{part: bone} for the rigid parts, "" or null filled with the bone nearest the part's centre."""
    if rigid is None or rigid == {}:
        return {}
    names = (
        [p["name"] for p in json.loads((src / "parts.json").read_text(encoding="utf-8"))["parts"]]
        if (src / "parts.json").is_file()
        else []
    )
    joints = json.loads((src / "skeleton.json").read_text(encoding="utf-8"))["joints"]
    known = {j["name"] for j in joints}
    if (
        not isinstance(rigid, dict)
        or not all(isinstance(b, str) or b is None for b in rigid.values())
        or set(rigid) - set(names)
        or set(rigid.values()) - known - {"", None}
    ):
        raise StepError(
            "bad_params",
            f'rigid {rigid}: {{part: bone or ""}} for the parts {names}, a joint of the skeleton',
        )
    bones = {part: bone for part, bone in rigid.items() if bone}
    if len(bones) < len(rigid):
        import trimesh

        s = trimesh.load(src / "mesh.glb", force="scene")
        for part in set(rigid) - set(bones):
            # the part's node and its children: trimesh hangs a part of two materials (two
            # primitives) under its node as children with generated names
            pts = [
                trimesh.transform_points(s.geometry[g].vertices, t)
                for n in s.graph.transforms.successors(part)
                if n in s.graph.nodes_geometry
                for t, g in [s.graph[n]]
            ]
            if not pts:
                raise StepError("failed", f"no mesh {part!r} in {src.name}/mesh.glb")
            bones[part] = _rig.nearest_bone(np.concatenate(pts).mean(0), joints)
    return bones


def make_rigid(glb: Path, bones: dict[str, str]) -> None:
    """Each part's mesh weighted fully to its bone, in place: JOINTS_0 → the bone, WEIGHTS_0 → 1."""
    doc, blob = _glb.read(glb)
    nodes = doc["nodes"]
    for part, bone in bones.items():
        node = next((n for n in nodes if n.get("name") == part and "mesh" in n), None)
        if node is None or "skin" not in node:
            raise RuntimeError(f"{glb.name}: no skinned mesh {part!r}")  # this engine's result only
        joints = doc["skins"][node["skin"]]["joints"]
        j = next((k for k, i in enumerate(joints) if nodes[i].get("name") == bone), None)
        if j is None:
            raise RuntimeError(f"{glb.name}: no bone {bone!r} in the skin of {part!r}")
        for prim in doc["meshes"][node["mesh"]]["primitives"]:
            att = prim["attributes"]
            jv, wv = (_glb.accessor(doc, blob, att[k]) for k in ("JOINTS_0", "WEIGHTS_0"))
            jv[:], wv[:] = 0, 0
            jv[:, 0] = j
            wv[:, 0] = 1 if wv.dtype.kind == "f" else np.iinfo(wv.dtype).max  # or normalized ints
            for k in ("JOINTS_0", "WEIGHTS_0"):
                doc["accessors"][att[k]].pop("min", None)  # stale now; optional for these
                doc["accessors"][att[k]].pop("max", None)
            for k in [k for k in att if k[:-1] in ("JOINTS_", "WEIGHTS_") and k[-1] != "0"]:
                del att[k]  # a 5th+ influence: gone, the bone has it all
    _glb.write(glb, doc, blob)


def run(op, in_dir, out_dir, params):
    src, out = Path(in_dir or ""), Path(out_dir)
    if not in_dir or not (src / "skeleton.json").is_file():
        raise StepError(
            "missing_input", "rig_skin needs a rig_fit version (Rig panel: Fit a skeleton)"
        )
    engine = params.get("engine", "auto")
    if engine not in ("auto", *ENGINES):
        raise StepError("bad_params", f"engine {engine!r}: auto, blender or skintokens")
    creature = _creature(src)
    if creature and engine == "skintokens":
        raise StepError(
            "bad_params", "SkinTokens knows the human skeleton only: use blender for a creature"
        )
    wanted = ["blender"] if creature else list(ENGINES) if engine == "auto" else [engine]
    bones = rigid_bones(src, params.get("rigid"))
    reports, notes = {}, {}
    for i, e in enumerate(wanted):
        text = {
            "blender": "bone heat on a closed proxy",
            "skintokens": "SkinTokens: the learned skin",
        }
        progress(0.1 + 0.4 * i, text[e])
        try:
            if e == "blender":
                extra = _blender(src, out, int(params.get("voxels", 120)))
            else:
                extra = _skintokens(src, out)
            # Blender's export drops TANGENT: the bake's frame back, or the normal map shades differently
            if (src / "mesh.glb").is_file():
                _glb.copy_tangents(src / "mesh.glb", out / f"rigged_{e}.glb")
            if bones:
                make_rigid(out / f"rigged_{e}.glb", bones)
            reports[e] = _qa(out, e, creature) | extra
        except (BlenderError, ImportError, FileNotFoundError, RuntimeError) as err:
            if engine != "auto":
                raise StepError("failed", f"{e}: {err}") from err
            notes[e] = str(err).splitlines()[0][:300]  # auto: the other engine still counts (M22: a
            # Blender bone-heat failure used to fail the step with a SkinTokens result in hand)
    if not reports:
        raise StepError("failed", "no skinning engine produced a result")
    picked = min(reports, key=lambda e: rank(reports[e]))
    shutil.copy(out / f"rigged_{picked}.glb", out / "rigged.glb")
    shutil.copy(src / "skeleton.json", out / "skeleton.json")
    if (src / "parts.json").is_file():
        shutil.copy(src / "parts.json", out / "parts.json")
    qa = {
        "picked": picked,
        "items": checklist(reports[picked]),
        "report": reports[picked],
        "engines": reports,
    }
    if notes:
        qa["skipped"] = notes
    loops = _loops(src)
    if loops is not None:
        qa["items"]["joint_loops"] = {
            "status": "pass" if not loops["few"] else "warn",
            "value": len(loops["few"]),
        }
        qa["loops"] = loops
    (out / "qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    outputs = {"rigged": "rigged.glb", "skeleton": "skeleton.json", "qa": "qa.json"}
    outputs |= {f"rigged_{e}": f"rigged_{e}.glb" for e in reports if e != picked}
    if (out / "parts.json").is_file():
        outputs["parts"] = "parts.json"
    rep = reports[picked]
    metrics = {"engine": picked, "fail": rank(rep)[0], "engines": sorted(reports)}
    skel = json.loads((out / "skeleton.json").read_text(encoding="utf-8"))
    metrics["template"] = skel.get("template", _templates.HUMAN)  # the Animate tab's clips (M36)
    metrics |= {
        k: rep[k] for k in ("unweighted_vertices", "max_influences", "pose_edge_stretch_max")
    }
    if "skintokens" in reports:
        metrics |= {
            f"skintokens_{k}": reports["skintokens"][k] for k in ("seconds", "peak_vram_gb")
        }
    if notes:
        metrics["skipped"] = notes
    if loops is not None:
        metrics["joints_few_loops"] = len(loops["few"])
    if bones:
        metrics["rigid"] = bones
    result = {"outputs": outputs, "metrics": metrics}
    if "skintokens" in reports:
        result |= {"models": [{"id": "VAST-AI/SkinTokens"}], "licenses": ["VAST-AI/SkinTokens"]}
    return result


if __name__ == "__main__":
    main(run)
