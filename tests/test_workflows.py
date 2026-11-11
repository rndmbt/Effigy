"""ComfyUI workflows as image models: the shipped manifests, the HTTP client against a fake ComfyUI, the edit
step, and what the server offers the UI."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import ClassVar

import pytest
import yaml
from PIL import Image

from server import comfyui, jobs, settings
from steps import _comfy, brief, edit
from steps._common import StepError


def test_shipped_workflows_bind_to_their_graphs():
    flows = _comfy.workflows()
    assert {
        "qwen21_turbo",
        "qwen21_turbo_edit",
        "moss",
        "moss_edit",
        "krea2_turbo",
        "anima_turbo",
    } <= set(flows)
    for name, m in flows.items():
        graph = json.loads(Path(m["file"]).read_text(encoding="utf-8"))
        need = {"prompt", "seed"} | ({"image"} if m["kind"] == "edit" else {"width", "height"})
        assert need <= set(m["set"]), name
        for target in m["set"].values():
            node, _, field = target.partition(".")
            assert field in graph[node]["inputs"], (name, target)
        assert m["models"], name
    # only a workflow whose every model is cleared stays unrestricted
    assert not flows["moss"]["restricted"]
    assert flows["qwen21_turbo"]["restricted"] and flows["moss_edit"]["restricted"]
    note = _comfy.provenance(flows["qwen21_turbo"])
    assert note["restricted"] and "Qwen/Qwen-Image-2.1" in note["note"]
    assert "restricted" not in _comfy.provenance(flows["moss"])


def test_own_workflow_without_models_is_restricted(tmp_path, monkeypatch):
    monkeypatch.setattr(_comfy, "FOLDERS", [tmp_path])
    (tmp_path / "mine.json").write_text("{}")
    (tmp_path / "mine.yaml").write_text(
        yaml.safe_dump({"kind": "image", "set": {"prompt": "1.text"}})
    )
    (tmp_path / "broken.yaml").write_text("kind: image")  # no graph beside it: not offered
    flows = _comfy.workflows()
    assert (
        list(flows) == ["mine"] and flows["mine"]["restricted"] and flows["mine"]["title"] == "mine"
    )
    with pytest.raises(StepError, match="not one of the edit workflows"):
        _comfy.manifest("mine", "edit")


class Fake(BaseHTTPRequestHandler):
    """Enough of ComfyUI's API for one prompt: it answers with a 8 x 8 picture and remembers the graph."""

    seen: ClassVar[list] = []
    free: ClassVar[list] = [1 * 2**30]  # /system_stats' answers, in turn (the last one stays)
    refuse = False

    def log_message(self, *args):
        pass

    def _send(self, body, code=200, kind="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path == "/prompt":
            if Fake.refuse:
                errors = {
                    "1": {
                        "class_type": "UNETLoader",
                        "errors": [{"message": "Value not in list", "details": "unet_name"}],
                    }
                }
                return self._send(
                    {
                        "error": {"message": "Prompt outputs failed validation"},
                        "node_errors": errors,
                    },
                    400,
                )
            Fake.seen.append(json.loads(body)["prompt"])
            return self._send({"prompt_id": "p1"})
        if self.path == "/upload/image":
            Fake.seen.append(("upload", b"PNG" in body))
            return self._send({"name": "effigy_input.png"})
        self._send({})

    def do_GET(self):
        if self.path.startswith("/history/"):
            image = {"filename": "a.png", "subfolder": "", "type": "temp"}
            return self._send(
                {
                    "p1": {
                        "status": {"status_str": "success"},
                        "outputs": {"13": {"images": [image]}},
                    }
                }
            )
        if self.path.startswith("/view"):
            buf = BytesIO()
            Image.new("RGB", (8, 8), "red").save(buf, "PNG")
            return self._send(buf.getvalue(), kind="image/png")
        free = Fake.free.pop(0) if len(Fake.free) > 1 else Fake.free[0]
        self._send({"devices": [{"vram_free": free}]})


@pytest.fixture
def fake(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("EFFIGY_COMFYUI", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(_comfy, "_shared_gb", lambda: None)
    Fake.seen, Fake.refuse = [], False
    yield Fake
    server.shutdown()


def test_generate_sets_the_graph_and_returns_the_pictures(fake):
    images, result = _comfy.generate("qwen21_turbo", "image", ["a cat", "a dog"], [7, 8], 768, 512)
    assert [im.size for im in images] == [(8, 8), (8, 8)]
    first, second = fake.seen
    assert first["5"]["inputs"]["prompt"] == "a cat" and second["5"]["inputs"]["prompt"] == "a dog"
    assert (first["10"]["inputs"]["noise_seed"], second["10"]["inputs"]["noise_seed"]) == (7, 8)
    assert (first["6"]["inputs"]["width"], first["6"]["inputs"]["height"]) == (768, 512)
    assert result["restricted"] and result["licenses"] == ["Qwen/Qwen-Image-2.1"]
    assert result["metrics"] == {"workflow": "qwen21_turbo", "shared_gb": None}


def test_a_refused_graph_names_the_node(fake):
    fake.refuse = True
    with pytest.raises(StepError, match="UNETLoader: Value not in list"):
        _comfy.generate("moss", "image", "a cat", [1], 1024, 1024)


def test_release_waits_for_the_memory_and_fails_when_it_stays(fake, monkeypatch):
    monkeypatch.setenv("EFFIGY_COMFYUI_FREE", str(15 * 2**30))
    fake.free = [1 * 2**30, 1 * 2**30, 15 * 2**30]
    _comfy.release(seconds=5)
    assert fake.free == [15 * 2**30]  # it looked until the memory was back
    fake.free = [3 * 2**30]
    with pytest.raises(StepError, match="kept 12.0 GB"):
        _comfy.release(seconds=0.5)


def test_edit_takes_the_chosen_picture_on_grey(fake, tmp_path):
    src, out = tmp_path / "v001_concepts", tmp_path / "out"
    src.mkdir(), out.mkdir()
    Image.new("RGBA", (16, 16), (0, 0, 0, 0)).save(src / "cand_2.png")
    meta = {
        "version": "v001_concepts",
        "step": "concepts",
        "outputs": {"cand_0": "cand_0.png", "cand_2": "cand_2.png"},
    }
    (src / "meta.json").write_text(json.dumps(meta))
    params = {"prompt": "add a hat", "workflow": "qwen21_turbo_edit", "key": "cand_2", "seed": 5}
    result = edit.run("run", src, out, params)
    assert fake.seen[0] == ("upload", True)
    graph = fake.seen[1]
    assert (
        graph["5"]["inputs"]["prompt"] == "add a hat"
        and graph["20"]["inputs"]["image"] == "effigy_input.png"
    )
    assert result["outputs"] == {"image": "image.png"} and (out / "image.png").is_file()
    assert result["restricted"] and result["metrics"]["seed"] == 5
    with pytest.raises(StepError, match="a prompt"):
        edit.run("run", src, out, {"workflow": "qwen21_turbo_edit"})


def test_brief_framing_replaces_the_template(tmp_path):
    brief.run(
        "run",
        None,
        tmp_path,
        {"prompt": "a lantern", "category": "prop", "framing": "side view, on white"},
    )
    assert (
        yaml.safe_load((tmp_path / "brief.yaml").read_text())["prompt_full"]
        == "a lantern, side view, on white"
    )
    brief.run("run", None, tmp_path, {"prompt": "a lantern", "category": "prop", "framing": ""})
    assert yaml.safe_load((tmp_path / "brief.yaml").read_text())["prompt_full"] == "a lantern"


def test_server_offers_workflows_only_with_comfyui_and_flags_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "FILE", tmp_path / "settings.json")
    monkeypatch.setattr(settings, "_cache", None)
    monkeypatch.setenv("EFFIGY_MODELS", str(tmp_path / "models"))
    assert comfyui.workflows() == []
    settings.write(comfyui_enabled=True, comfyui_models="")
    monkeypatch.setattr(comfyui, "installed", lambda: True)
    (tmp_path / "models/vae").mkdir(parents=True)
    (tmp_path / "models/vae/qwen_image_vae.safetensors").write_text("")
    flows = {w["id"]: w for w in comfyui.workflows()}
    assert flows["anima_turbo"]["missing"] == [
        "anima-turbo-v1.1.safetensors",
        "qwen_3_06b_base.safetensors",
    ]
    assert flows["moss"]["kind"] == "image" and not flows["moss"]["restricted"]


def test_a_workflow_job_is_comfyuis():
    assert jobs._workflow(jobs.Job("a", "concepts", "v1", {"workflow": "moss"})) == "moss"
    assert jobs._workflow(jobs.Job("a", "edit", "v1", {"workflow": "moss_edit"})) == "moss_edit"
    assert jobs._workflow(jobs.Job("a", "concepts", "v1", {})) is None
    assert jobs._workflow(jobs.Job("a", "trellis2", "v1", {"workflow": "x"})) is None
