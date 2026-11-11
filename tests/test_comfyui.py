"""ComfyUI setup keeps native model paths unchanged and gives its optional folder priority."""

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from server import comfyui, comfyui_install, settings
from server.app import app


def test_setup_and_model_folders(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "FILE", tmp_path / "settings.json")
    monkeypatch.setattr(settings, "_cache", None)
    monkeypatch.setattr(comfyui, "HOME", tmp_path / "comfyui")
    monkeypatch.setattr(comfyui, "python", lambda: Path(sys.executable))
    monkeypatch.setenv("EFFIGY_MODELS", str(tmp_path / "native"))
    client = TestClient(app)
    assert client.get("/api/setup/comfyui").json()["download_path"] == str(tmp_path / "native")
    assert client.post("/api/setup/comfyui/install").status_code == 409
    assert (
        client.post("/api/setup/comfyui", json={"enabled": True, "models": "relative"}).status_code
        == 400
    )
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "keep.txt").write_text("existing model store")
    state = client.post("/api/setup/comfyui", json={"enabled": True, "models": str(shared)}).json()
    assert state["download_path"] == str(shared)
    assert settings.models_root() == tmp_path / "native"
    comfyui.prepare()
    paths = yaml.safe_load((comfyui.HOME / "extra_model_paths.yaml").read_text())
    assert paths["effigy"]["base_path"] == str(tmp_path / "native")
    assert "is_default" not in paths["effigy"]
    (comfyui.HOME / "source").mkdir()
    (comfyui.HOME / "source/main.py").touch()
    (comfyui.HOME / "installed.json").write_text(json.dumps({"revision": comfyui.REVISION}))
    assert comfyui.installed()
    cmd = comfyui.command()
    assert cmd[cmd.index("--models-directory") + 1] == str(shared)
    assert comfyui.environment()["HF_HUB_CACHE"] == str(shared / "huggingface")
    assert comfyui.environment()["TORCH_HOME"] == str(shared / "torch")
    assert (shared / "keep.txt").read_text() == "existing model store"
    comfyui.choose(True, "")
    comfyui.prepare()
    assert comfyui.model_root() == tmp_path / "native"
    assert yaml.safe_load((comfyui.HOME / "extra_model_paths.yaml").read_text()) == {}
    assert comfyui.choose(False, str(tmp_path / "missing"))["enabled"] is False
    assert not (tmp_path / "missing").exists()


def test_source_is_pinned_and_installer_is_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr(comfyui, "HOME", tmp_path / "comfyui")
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(f"ComfyUI-{comfyui.REVISION}/main.py", "# source archive test fixture")
    monkeypatch.setattr(comfyui_install, "SOURCE_URL", archive.as_uri())
    monkeypatch.setattr(comfyui_install, "SOURCE_SHA", "0" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        comfyui_install.source()
    monkeypatch.setattr(
        comfyui_install, "SOURCE_SHA", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    tree = comfyui_install.source()
    assert (tree / "main.py").is_file()
    assert comfyui_install.source() == tree
    calls = []
    monkeypatch.setattr(comfyui_install, "install", lambda: calls.append("install"))
    monkeypatch.setattr(settings, "_cache", {})
    comfyui_install.main(["--if-enabled"])
    assert not calls
    monkeypatch.setattr(settings, "_cache", {"comfyui_enabled": True})
    comfyui_install.main(["--if-enabled"])
    assert calls == ["install"]
