"""Generate 3D selects the optional backend without changing the library contract."""

import json
import threading
from pathlib import Path

import pytest

from server import comfyui, jobs, library, presets, settings
from steps import trellis2


@pytest.mark.parametrize("enabled", [False, True])
def test_generate_backend_is_pinned_at_submission(tmp_path, monkeypatch, enabled):
    monkeypatch.setattr(library, "asset_dir", lambda _: tmp_path)
    monkeypatch.setattr(settings, "_cache", {"comfyui_enabled": enabled})
    monkeypatch.setattr(threading.Thread, "start", lambda _: None)
    job, _ = jobs.submit("backend_test", "trellis2", None, {"weights": "faithful"})
    assert job.params["backend"] == ("comfyui" if enabled else "native")
    assert bool(jobs._workflow(job)) is enabled
    monkeypatch.setattr(settings, "_cache", {"comfyui_enabled": not enabled})
    assert bool(jobs._workflow(job)) is enabled
    texture, _ = jobs.submit("backend_test", "trellis2", None, {"op": "texture"})
    assert jobs._workflow(texture) is None


def test_comfy_generate_does_not_load_native_pipeline(tmp_path, monkeypatch):
    from steps import _comfy3d

    (tmp_path / "image_rgb.png").touch()
    called = []
    monkeypatch.setattr(_comfy3d, "run", lambda *args: called.append(args) or {"outputs": {}})
    monkeypatch.setattr(trellis2, "pipeline", lambda _: pytest.fail("loaded native generator"))
    trellis2.run("generate", tmp_path, tmp_path, {"backend": "comfyui"})
    assert len(called) == 1


def test_comfy_generation_graph_has_quality_and_release_stages():
    graph = json.loads((Path(__file__).parents[1] / "workflows/pixal3d_generate.json").read_text())
    kinds = {n["class_type"] for n in graph.values()}
    assert {
        "MoGeInference",
        "Pixal3DConditioning",
        "RemeshMesh",
        "EffigyExport3D",
        "EffigyUnload",
    } <= kinds
    assert "UnwrapMesh" not in kinds  # UVgami, not the template's PEC unwrap
    assert sum(n["class_type"] == "EffigyUnload" for n in graph.values()) >= 5


def test_graph_camera_binding_and_pruning(tmp_path, monkeypatch):
    from PIL import Image

    from steps import _comfy3d

    Image.new("RGBA", (32, 32), "grey").save(tmp_path / "image.png")
    monkeypatch.setattr(_comfy3d._comfy, "_upload", lambda _: "input.png")
    p = trellis2._params(
        presets.PRESETS["trellis2"]["faithful"]
    )  # a preset's fov does not switch MoGe off
    g, _ = _comfy3d.graph(tmp_path, p)
    assert g["298"]["inputs"]["camera_angle_x"] == ["unload_camera", 0]
    assert 0 < g["241"]["inputs"]["drop_small_components"] <= 0.001  # bubbles go, real parts stay
    g, _ = _comfy3d.graph(tmp_path, p | {"camera_auto": False, "fov": 0.2})
    assert "56" not in g
    assert g["298"]["inputs"]["camera_angle_x"] > 0.2  # UI node takes degrees, step takes radians
    g, _ = _comfy3d.graph(tmp_path, p | {"weights": "fast", "pipeline_type": "512"})
    assert "56" not in g and "94" not in g and "23" not in g
    assert g["92"]["inputs"]["samples"] == ["unload_shape_sample", 0]


def test_converted_models_use_selected_store(tmp_path, monkeypatch):
    from server import models

    monkeypatch.setattr(settings, "_cache", {"comfyui_models": str(tmp_path)})
    entries = [e for e in models.entries() if "comfy3d" in e["groups"]]
    assert entries
    for e in entries:
        for rel in e["filters"]["allow_patterns"]:
            target = models.target(e, rel)
            assert target == tmp_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
            assert target in models._files(e)


def test_selected_missing_backend_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setattr(settings, "_cache", {"comfyui_enabled": True})
    monkeypatch.setattr(comfyui, "installed", lambda: False)
    with pytest.raises(Exception, match="install ComfyUI"):
        comfyui.start()
