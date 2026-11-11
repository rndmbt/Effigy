"""Learned material channel semantics and texture-workflow selection without model weights."""

import json

import numpy as np
import pytest
from PIL import Image

from server import jobs
from steps import _pbr, refine, textures


def test_linear_material_predictions_are_not_gamma_corrected():
    # Build the estimator around a tiny fake pipeline: scalar .25 must become 64, not a display-gamma value.
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA generator required")
    e = object.__new__(_pbr.Estimator)
    e.seed, e.cache = 42, {}

    class Pipe:
        def __call__(self, *args, **kwargs):
            from types import SimpleNamespace

            a = np.full((2, 4, 4, 3), 0.25, dtype=np.float32)
            a[1, ..., 1] = 0.75
            return SimpleNamespace(prediction=a)

    e.pipe = Pipe()
    maps = e.predict(Image.new("RGB", (4, 4)))
    assert np.asarray(maps["roughness"]).tolist() == [[64] * 4] * 4
    assert np.asarray(maps["metallic"]).tolist() == [[191] * 4] * 4
    assert maps["basecolor"].getpixel((0, 0)) == (64, 64, 64)


def test_pbr_replaces_material_channels_but_preserves_normal_and_ao(tmp_path, monkeypatch):
    src, out = tmp_path / "001_textures", tmp_path / "002_textures"
    src.mkdir()
    out.mkdir()
    from steps import _textures as tx

    original = {
        "basecolor": Image.new("RGBA", (64, 64), (150, 100, 50, 170)),
        "normal": Image.new("RGB", (64, 64), (128, 128, 255)),
        "ao": Image.new("L", (64, 64), 190),
    }
    outputs = tx.write_set(src, original, {"settings": {}})
    (src / "meta.json").write_text(json.dumps({"outputs": outputs, "step": "textures"}))
    generated = {
        "basecolor": Image.new("RGB", (64, 64), (200, 140, 100)),
        "roughness": Image.new("L", (64, 64), 90),
        "metallic": Image.new("L", (64, 64), 30),
    }
    monkeypatch.setattr(_pbr, "estimate", lambda *args: (generated, {"method": _pbr.MODEL}))
    result = textures.run("pbr", src, out, {})
    (out / "meta.json").write_text(json.dumps({"outputs": result["outputs"], "step": "textures"}))
    after, manifest = tx.load_set(out, 0)
    for role in ("ao", "normal"):
        np.testing.assert_array_equal(after[role], original[role])
    assert after["basecolor"].getchannel("A").getextrema() == (170, 170)
    assert after["roughness"].getextrema() == (90, 90)
    assert result["licenses"] == [_pbr.MODEL]
    assert manifest["settings"]["roughnessFactor"] == 1


def test_texture_jobs_use_the_workflow_runner():
    assert (
        jobs._workflow(jobs.Job("a", "refine", "001_bake", {"workflow": "qwen21_texture"}))
        == "qwen21_texture"
    )
    assert (
        jobs._workflow(
            jobs.Job("a", "textures", "001_bake", {"op": "enhance", "workflow": "moss_edit"})
        )
        == "moss_edit"
    )
    assert jobs._workflow(jobs.Job("a", "textures", "001_bake", {"op": "pbr"})) is None


def test_comfy_texture_generator_requires_a_reference_for_reference_workflow(tmp_path):
    src = tmp_path / "001_bake"
    src.mkdir()
    (src / "meta.json").write_text(json.dumps({"step": "bake", "outputs": {}}))
    with pytest.raises(Exception, match="original prep reference"):
        refine._comfy_generator("qwen21_texture", src)
