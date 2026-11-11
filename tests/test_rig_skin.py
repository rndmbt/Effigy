"""rig_skin engine auto: a failing engine is a note when the other one delivers (M22); rigid parts
(M32) around faked engines."""

import json

import numpy as np
import pytest

from steps import rig_skin
from steps._blender import BlenderError
from steps._common import StepError

REPORT = {
    "unweighted_vertices": 0,
    "weight_sum_off": 0,
    "max_influences": 4,
    "lr_asymmetry": 0.05,
    "pose_volume_change_pct_max": 1.0,
    "pose_edge_stretch_max": 1.5,
}


def test_auto_keeps_the_engine_that_worked(tmp_path, monkeypatch):
    src, out = tmp_path / "100_rig_fit", tmp_path / "110_rig_skin"
    src.mkdir()
    out.mkdir()
    (src / "skeleton.json").write_text("{}")

    def heat_fails(s, o, voxels):
        raise BlenderError("bone heat found no solution at any proxy resolution")

    def skintokens(s, o):
        (o / "rigged_skintokens.glb").write_bytes(b"glb")
        return {"seconds": 1.0, "peak_vram_gb": 3.4}

    monkeypatch.setattr(rig_skin, "_blender", heat_fails)
    monkeypatch.setattr(rig_skin, "_skintokens", skintokens)
    monkeypatch.setattr(rig_skin, "_qa", lambda o, e, creature: dict(REPORT))
    res = rig_skin.run("run", str(src), str(out), {"engine": "auto"})
    qa = json.loads((out / "qa.json").read_text())
    assert res["metrics"]["engine"] == "skintokens" and (out / "rigged.glb").is_file()
    assert "bone heat" in qa["skipped"]["blender"] and res["metrics"]["skipped"] == qa["skipped"]


def parts_rig(tmp_path):
    """A rig_fit version of a parts version: parts.json beside the skeleton."""
    src = tmp_path / "100_rig_fit"
    src.mkdir()
    joints = [{"name": "Hips", "parent": None, "position": [0, 1, 0]}]
    joints.append({"name": "Head", "parent": "Hips", "position": [0, 1.6, 0]})
    (src / "skeleton.json").write_text(json.dumps({"joints": joints}))
    pj = {"views": "x", "labels": [], "parts": [{"id": 1, "name": "Helmet", "faces": 3}]}
    (src / "parts.json").write_text(json.dumps(pj))
    return src


def test_rigid_parts_go_into_every_engines_glb_before_the_qa(tmp_path, monkeypatch):
    src, out = parts_rig(tmp_path), tmp_path / "110_rig_skin"
    out.mkdir()
    order = []

    def engine(name):
        def run(s, o, *_):
            (o / f"rigged_{name}.glb").write_bytes(b"glb")
            order.append(("skin", name))
            return {"seconds": 1.0, "peak_vram_gb": 3.4}

        return run

    monkeypatch.setattr(rig_skin, "_blender", engine("blender"))
    monkeypatch.setattr(rig_skin, "_skintokens", engine("skintokens"))
    monkeypatch.setattr(rig_skin, "make_rigid", lambda glb, b: order.append(("rigid", glb.name, b)))
    monkeypatch.setattr(
        rig_skin, "_qa", lambda o, e, creature: order.append(("qa", e)) or dict(REPORT)
    )
    res = rig_skin.run("run", str(src), str(out), {"rigid": {"Helmet": "Head"}})
    assert order == [
        ("skin", "blender"),
        ("rigid", "rigged_blender.glb", {"Helmet": "Head"}),
        ("qa", "blender"),
        ("skin", "skintokens"),
        ("rigid", "rigged_skintokens.glb", {"Helmet": "Head"}),
        ("qa", "skintokens"),
    ]
    assert res["metrics"]["rigid"] == {"Helmet": "Head"} and res["outputs"]["parts"] == "parts.json"
    assert (out / "parts.json").read_text() == (src / "parts.json").read_text()


@pytest.mark.parametrize(
    "rigid",
    [
        {"Body": "Head"},
        {"Shield": ""},
        {"Helmet": "Tail"},
        ["Helmet"],  # not a map
        [],
        "",
        {"Helmet": ["Head"]},  # a bone that is not a name
    ],  # Body is the rest; no such part/bone
)
def test_rigid_bad_params(tmp_path, rigid):
    with pytest.raises(StepError) as e:
        rig_skin.rigid_bones(parts_rig(tmp_path), rigid)
    assert e.value.code == "bad_params"


def test_copy_tangents_gives_a_blender_export_the_bakes_frame_back(tmp_path):
    pytest.importorskip("torch")  # the bake writer
    import trimesh

    from steps import _glb
    from steps._bake import write_glb
    from steps._mesh import smooth_normals

    s = trimesh.creation.icosphere(2)
    uv = (s.vertices[:, :2] + 1) / 2
    px = np.full((4, 4, 3), 128, np.uint8)
    write_glb(
        tmp_path / "baked.glb",
        s.vertices,
        s.faces,
        uv,
        np.full((4, 4, 4), 255, np.uint8),
        px,
        smooth_normals(s.vertices, s.faces),  # as bake writes them
        px,
    )
    doc, blob = _glb.read(tmp_path / "baked.glb")
    att = doc["meshes"][0]["primitives"][0]["attributes"]
    want = _glb.accessor(doc, blob, att.pop("TANGENT")).copy()
    _glb.write(tmp_path / "skinned.glb", doc, blob)  # as Blender writes it: no TANGENT
    assert _glb.copy_tangents(tmp_path / "baked.glb", tmp_path / "skinned.glb") == 1
    doc, blob = _glb.read(tmp_path / "skinned.glb")
    got = _glb.accessor(doc, blob, doc["meshes"][0]["primitives"][0]["attributes"]["TANGENT"])
    assert np.allclose(got, want)
    assert _glb.copy_tangents(tmp_path / "baked.glb", tmp_path / "skinned.glb") == 0  # has one now
