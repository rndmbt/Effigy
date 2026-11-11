"""steps/trellis2.py end to end on the CPU with the two GPU pieces faked (the model, CuMesh)."""

import json
import math
from typing import ClassVar

import numpy as np
import pytest

torch = pytest.importorskip("torch")
trimesh = pytest.importorskip("trimesh")
from PIL import Image

from steps import trellis2
from steps._common import load_ovoxel


def box(lo, hi):
    m = trimesh.creation.box(bounds=[lo, hi])
    return torch.tensor(m.vertices, dtype=torch.float32), torch.tensor(m.faces, dtype=torch.int32)


class FakeMesh:
    def __init__(self):
        self.vertices, self.faces = box([-0.2, -0.1, -0.4], [0.2, 0.1, 0.35])  # TRELLIS frame, Z up
        g = torch.stack(torch.meshgrid(*[torch.arange(16)] * 3, indexing="ij"), -1).reshape(-1, 3)
        self.coords = g.int()
        z = g[:, 2:3].float() / 15  # colour ramps up the Z (up) axis
        self.attrs = torch.cat(
            [
                z,
                1 - z,
                torch.full_like(z, 0.5),
                torch.zeros_like(z),
                torch.full_like(z, 0.7),
                torch.ones_like(z),
            ],
            1,
        )
        self.voxel_size = 1 / 16
        self.layout = {
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        }


class FakePipe:
    calls: ClassVar[list] = []

    def run(self, image, **kw):
        self.calls.append(kw)
        return [FakeMesh()]


def per_face_atlas(vertices, faces, target):
    """CPU stand-in for CuMesh: no simplification, one atlas cell per triangle."""
    f = faces.long()
    lv, lf = vertices[f].reshape(-1, 3), torch.arange(3 * len(f)).reshape(-1, 3)
    n = math.ceil(math.sqrt(len(f)))
    cell = torch.arange(len(f))
    x0, y0, m = (cell % n).float() / n, (cell // n).float() / n, 0.1 / n
    uv = torch.stack(
        [
            torch.stack([x0 + m, y0 + m], 1),
            torch.stack([x0 + 1 / n - m, y0 + m], 1),
            torch.stack([x0 + m, y0 + 1 / n - m], 1),
        ],
        1,
    ).reshape(-1, 2)
    normals = torch.nn.functional.normalize(
        torch.cross(lv[1::3] - lv[0::3], lv[2::3] - lv[0::3], dim=1), dim=1
    ).repeat_interleave(3, 0)
    return lv, lf, uv, normals, None, (vertices, faces)


def test_generate_writes_every_output(tmp_path, monkeypatch):
    monkeypatch.setattr(trellis2, "pipeline", lambda weights: FakePipe())
    monkeypatch.setattr(trellis2, "_simplify_unwrap", per_face_atlas)
    monkeypatch.delenv(
        "EFFIGY_BLENDER", raising=False
    )  # no Blender: the preview is skipped, not fatal
    prep, out = tmp_path / "020_prep", tmp_path / "030_trellis2"
    prep.mkdir()
    out.mkdir()
    Image.new("RGB", (64, 64), (90, 90, 90)).save(prep / "image_rgb.png")
    res = trellis2.run(
        "generate",
        str(prep),
        str(out),
        {"seed": 3, "steps": 8, "texture_size": 256, "faces": 1000, "height_m": 1.5},
    )

    assert res["outputs"] == {
        "mesh": "hi.glb",
        "ovoxel": "ovoxel.npz",
        "hi_trellis": "hi_trellis.ply",
    }
    assert FakePipe.calls[-1]["seed"] == 3 and FakePipe.calls[-1]["preprocess_image"] is False
    assert FakePipe.calls[-1]["shape_slat_sampler_params"] == {"steps": 8}
    vol = load_ovoxel(out / "ovoxel.npz")
    assert (
        vol["resolution"] == 16 and len(vol["coords"]) == 16**3 and vol["layout"]["alpha"] == [5, 6]
    )

    glb = trimesh.load(out / "hi.glb", force="mesh")
    lo, hi = glb.bounds
    np.testing.assert_allclose(
        [lo[1], hi[1] - lo[1]], [0, 1.5], atol=1e-5
    )  # base at y = 0, 1.5 m tall
    np.testing.assert_allclose(
        (lo[[0, 2]] + hi[[0, 2]]) / 2, [0, 0], atol=1e-5
    )  # centred on the origin
    tex = np.asarray(glb.visual.material.baseColorTexture)
    assert tex.shape[:2] == (256, 256) and tex[..., 3].min() == 255
    frame = res["metrics"]["frame"]["to_library_from_source"]
    assert (
        len(frame) == 4
        and res["metrics"]["voxels"] == 16**3
        and res["licenses"][0] == "microsoft/TRELLIS.2-4B"
    )

    # the top of the box (TRELLIS z = 0.35) must be red-ish, the bottom (z = -0.4) green-ish: frame + bake agree
    uv, faces = glb.visual.uv, glb.faces
    tops = [f for f in faces if np.allclose(glb.vertices[f][:, 1], hi[1])]
    bottoms = [f for f in faces if np.allclose(glb.vertices[f][:, 1], lo[1])]

    def colour(f):  # sample the texture at the triangle's UV centroid (trimesh UVs: v up)
        u, v = uv[f].mean(0)
        return tex[int((1 - v) * 256), int(u * 256), :3].astype(int)

    assert colour(tops[0])[0] > colour(tops[0])[1] and colour(bottoms[0])[1] > colour(bottoms[0])[0]
    json.dumps(res)  # RESULT must be JSON-serialisable


def test_faithful_passes_the_camera_and_uses_pixal3ds_frame(tmp_path, monkeypatch):
    FakePipe.calls.clear()
    monkeypatch.setattr(trellis2, "pipeline", lambda weights: FakePipe())
    monkeypatch.setattr(trellis2, "_simplify_unwrap", per_face_atlas)
    monkeypatch.delenv("EFFIGY_BLENDER", raising=False)
    prep, out = tmp_path / "020_prep", tmp_path / "030_trellis2"
    prep.mkdir()
    out.mkdir()
    Image.new("RGB", (64, 64), (90, 90, 90)).save(prep / "image_rgb.png")
    params = {"weights": "faithful", "fov": 0.3, "texture_size": 256, "faces": 1000}
    res = trellis2.run("generate", str(prep), str(out), params)

    cam = FakePipe.calls[-1]["camera_params"]
    assert cam["camera_angle_x"] == 0.3 and cam["distance"] == pytest.approx(0.5 / math.tan(0.15))
    assert res["licenses"][0] == "TencentARC/Pixal3D" and res["metrics"]["weights"] == "faithful"
    lo, hi = trimesh.load(out / "hi.glb", force="mesh").bounds
    # the fake box spans y ∈ [-0.1, 0.1] (Pixal3D's up axis): 0.2 tall → scaled to 1 m
    np.testing.assert_allclose(hi - lo, [0.4 / 0.2, 1.0, 0.75 / 0.2], atol=1e-5)

    with pytest.raises(trellis2.StepError, match="bad trellis2 params"):
        trellis2._params({"weights": "faithful", "pipeline_type": "512"})  # cascades only


class FakeTexturing:
    """Stands in for Trellis2TexturingPipeline: voxels on the box's surface, coloured by height (TRELLIS z)."""

    pbr_attr_layout = FakeMesh().layout
    models: ClassVar[dict] = {
        "tex_slat_flow_model_512": "flow512",
        "tex_slat_flow_model_1024": "flow1024",
    }

    def get_cond(self, images, res):
        self.cond = (images[0].size, res)
        return {}

    def encode_shape_slat(self, mesh, res):
        self.mesh = mesh
        return "shape"

    def sample_tex_slat(self, cond, flow, shape, params):
        self.flow, self.params = flow, params
        return "tex"

    def decode_tex_slat(self, slat):
        res = 512
        lo, hi = np.floor((self.mesh.bounds + 0.5) * res).astype(int)
        slabs = []
        for axis in range(3):  # the six faces of the box, one voxel layer each
            others = [np.arange(lo[a], hi[a] + 1) for a in range(3) if a != axis]
            grid = np.stack(np.meshgrid(*others, indexing="ij"), -1).reshape(-1, 2)
            for side in (lo[axis], hi[axis]):
                c = np.insert(grid, axis, side, axis=1)
                slabs.append(c)
        xyz = torch.tensor(np.unique(np.concatenate(slabs), axis=0))
        z = (xyz[:, 2:3].float() - lo[2]) / (hi[2] - lo[2])
        feats = torch.cat(
            [
                z,
                1 - z,
                torch.full_like(z, 0.5),
                torch.zeros_like(z),
                torch.full_like(z, 0.7),
                torch.ones_like(z),
            ],
            1,
        )

        class Vox:
            coords = torch.cat([torch.zeros(len(xyz), 1, dtype=torch.long), xyz], 1)

        Vox.feats = feats
        return Vox


def test_texture_keeps_mesh_and_uvs(tmp_path, monkeypatch):
    from steps import bake
    from steps._bake import write_glb

    fake = FakeTexturing()
    monkeypatch.setattr(trellis2, "pipeline", lambda weights: fake)
    asset = tmp_path / "crate"
    for name, parent, outputs, licenses in [
        ("020_prep", None, {"image_rgb": "image_rgb.png"}, ["Tongyi-MAI/Z-Image-Turbo"]),
        ("030_trellis2", "020_prep", {"mesh": "hi.glb"}, ["microsoft/TRELLIS.2-4B"]),
        ("040_blender_pull", "030_trellis2", {"mesh": "mesh.glb"}, ["microsoft/TRELLIS.2-4B"]),
    ]:
        (asset / name).mkdir(parents=True)
        meta = {"version": name, "parent": parent, "outputs": outputs, "licenses": licenses}
        (asset / name / "meta.json").write_text(json.dumps(meta))
    Image.new("RGB", (64, 64), (90, 90, 90)).save(asset / "020_prep" / "image_rgb.png")
    # a pulled mesh in the library frame (Y up, metres), UVs one atlas cell per triangle
    v, f = box([-0.3, 0.0, -0.2], [0.3, 1.2, 0.2])
    lv, lf, uv, normals, _, _ = per_face_atlas(v, f, 0)
    flat = np.full((4, 4, 4), 128, np.uint8)
    write_glb(
        asset / "040_blender_pull" / "mesh.glb",
        lv.numpy(),
        lf.numpy(),
        uv.numpy(),
        flat,
        flat[..., :3],
        normals.numpy(),
    )
    before = trimesh.load(asset / "040_blender_pull" / "mesh.glb", force="mesh", process=False)

    out = tmp_path / "out"
    out.mkdir()
    res = trellis2.run(
        "texture",
        str(asset / "040_blender_pull"),
        str(out),
        {"resolution": 512, "texture_size": 256, "seed": 1},
    )

    t = fake.mesh.bounds
    assert np.all(t >= -0.5) and np.all(t <= 0.5) and np.isclose((t[1] - t[0]).max(), 0.99999)
    assert np.isclose(t[1, 2] - t[0, 2], 0.99999)  # the 1.2 m Y-up height is the model's Z
    assert fake.flow == "flow512" and fake.params == {"steps": 12} and fake.cond[1] == 512
    after = trimesh.load(out / "mesh.glb", force="mesh", process=False)
    np.testing.assert_array_equal(after.faces, before.faces)
    np.testing.assert_allclose(after.vertices, before.vertices, atol=1e-6)
    np.testing.assert_allclose(after.visual.uv, before.visual.uv, atol=1e-7)  # UVs untouched
    tex = np.asarray(after.visual.material.baseColorTexture)
    assert tex.shape[:2] == (256, 256) and res["metrics"]["coverage"] > 0.2

    def colour(face):  # texel at the triangle's UV centroid (trimesh UVs: v up)
        u, w = after.visual.uv[face].mean(0)
        return tex[int((1 - w) * 256), int(u * 256), :3].astype(int)

    top = next(fc for fc in after.faces if np.allclose(after.vertices[fc][:, 1], 1.2))
    bottom = next(fc for fc in after.faces if np.allclose(after.vertices[fc][:, 1], 0.0))
    assert colour(top)[0] > colour(top)[1] and colour(bottom)[1] > colour(bottom)[0]
    assert res["licenses"] == sorted(
        {"Tongyi-MAI/Z-Image-Turbo", "microsoft/TRELLIS.2-4B", trellis2.DINOV3}
    )
    assert res["metrics"]["image"] == "020_prep"

    # M3 re-bake finds this version's volume and maps the mesh back onto it
    (out / "meta.json").write_text(
        json.dumps(
            {
                "version": "050_trellis2",
                "parent": "040_blender_pull",
                "outputs": res["outputs"],
                "frame": res["metrics"]["frame"],
            }
        )
    )
    vol_dir, frame = bake.find_volume(out)
    assert vol_dir == out
    hi = trimesh.load(out / "hi_trellis.ply", process=False)
    np.testing.assert_allclose(
        before.vertices @ np.linalg.inv(frame)[:3, :3].T + np.linalg.inv(frame)[:3, 3],
        hi.vertices,
        atol=1e-5,
    )

    with pytest.raises(trellis2.StepError, match="no version"):
        trellis2.run("texture", str(asset / "040_blender_pull"), str(out), {"image": "../x"})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuBVH needs CUDA")
def test_face_outward_turns_inward_faces():
    import trimesh

    from steps._mesh import face_outward

    box = trimesh.creation.box().subdivide().subdivide()
    faces = box.faces.copy()
    faces[::2] = faces[::2, ::-1]  # half the faces wound inward

    class Mesh:  # the two CuMesh calls face_outward uses
        v = torch.tensor(box.vertices, dtype=torch.float32).cuda()
        f = torch.tensor(faces, dtype=torch.int32).cuda()

        def read(self):
            return self.v, self.f

        def init(self, v, f):
            self.v, self.f = v, f

    m = Mesh()
    face_outward(m)
    out = trimesh.Trimesh(box.vertices, m.f.cpu().numpy(), process=False)
    assert ((out.face_normals * out.triangles_center).sum(1) > 0).all()  # every face faces out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CuMesh needs CUDA")
def test_make_manifold_drops_faces_on_overshared_edges():
    import cumesh

    from steps._mesh import make_manifold, mesh_report

    # a fin: three triangles on the edge 0-1, plus a quad that's fine
    v = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [3, 0, 0],
            [4, 0, 0],
            [3, 1, 0],
            [4, 1, 0],
        ],
        dtype=torch.float32,
    )
    f = torch.tensor([[0, 1, 2], [1, 0, 3], [0, 1, 4], [5, 6, 7], [6, 8, 7]], dtype=torch.int32)
    m = cumesh.CuMesh()
    m.init(v.cuda(), f.cuda())
    make_manifold(m)
    v2, f2 = (t.cpu().numpy() for t in m.read())
    assert mesh_report(v2, f2)["non_manifold_edges"] == 0 and len(f2) >= 4


def test_a_failure_inside_the_models_evicts_the_resident_pipeline(tmp_path, monkeypatch):
    prep, out = tmp_path / "020_prep", tmp_path / "030_trellis2"
    prep.mkdir()
    out.mkdir()
    Image.new("RGB", (64, 64), (90, 90, 90)).save(prep / "image_rgb.png")
    monkeypatch.setattr(trellis2, "PIPES", {"faithful": object()})

    def oom(weights):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (injected)")

    monkeypatch.setattr(trellis2, "pipeline", oom)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        trellis2.run("generate", str(prep), str(out), {"seed": 1})
    assert trellis2.PIPES == {}  # the next request loads into a clean card

    trellis2.PIPES["faithful"] = (
        object()
    )  # a bad input never reached the models: the pipeline stays
    with pytest.raises(Exception, match="generate needs a prep version"):
        trellis2.run("generate", str(tmp_path / "missing"), str(out), {})
    assert "faithful" in trellis2.PIPES


def test_auto_background_is_grey_for_both_generators(tmp_path):
    """The models see the cut-out on a flat background; black hides black fur (a black cat came out without its
    back with Fast): auto uses grey for both generators, and an explicit choice wins."""
    rgba = np.zeros((40, 40, 4), np.uint8)
    rgba[10:30, 10:30] = (10, 10, 10, 255)  # a black square: all of the subject near black
    Image.fromarray(rgba, "RGBA").save(tmp_path / "image.png")
    img, bg = trellis2._framed(tmp_path, "auto", "fast")
    assert bg == {"background": "grey", "dark_share": 1.0}
    assert np.asarray(img)[0, 0].tolist() == [128, 128, 128] and np.asarray(img)[
        20, 20
    ].tolist() == [10, 10, 10]
    assert trellis2._framed(tmp_path, "auto", "faithful")[1]["background"] == "grey"
    assert trellis2._framed(tmp_path, "white", "faithful")[1]["background"] == "white"
    rgba[10:30, 10:30] = (200, 180, 160, 255)  # a light subject
    Image.fromarray(rgba, "RGBA").save(tmp_path / "image.png")
    assert trellis2._framed(tmp_path, "auto", "fast")[1] == {
        "background": "grey",
        "dark_share": 0.0,
    }
    (tmp_path / "image.png").unlink()  # a prep from before the option: its black composite
    Image.new("RGB", (8, 8)).save(tmp_path / "image_rgb.png")
    assert trellis2._framed(tmp_path, "auto", "fast")[1] == {"background": "black"}
    with pytest.raises(Exception, match="bad trellis2 params"):
        trellis2._params({"background": "pink"})


def test_no_op_means_generate(tmp_path):
    """The owner's agent (2026-10-11): a request without `op` failed as "unknown op 'run'"; it generates."""
    from steps._common import StepError

    with pytest.raises(StepError) as e:
        trellis2._run(
            "run", tmp_path, tmp_path, {"weights": "fast"}
        )  # no image: fails later, not on the op
    assert e.value.code != "bad_params" or "op" not in str(e.value)
