"""Optional ComfyUI installation and model paths; ComfyUI runs in its own environment."""

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import yaml

from server import jobs, presets, settings
from server.library import ApiError
from tools.download_models import PINNED

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / "comfyui"
REVISION = "2c75dc1b379a59e0880619d029607af5169a62ef"  # ComfyUI v0.44.0
MODEL_FOLDERS = [
    "checkpoints",
    "configs",
    "loras",
    "vae",
    "clip_vision",
    "style_models",
    "embeddings",
    "diffusers",
    "vae_approx",
    "controlnet",
    "gligen",
    "upscale_models",
    "latent_upscale_models",
    "hypernetworks",
    "photomaker",
    "classifiers",
    "model_patches",
    "audio_encoders",
    "background_removal",
    "geometry_estimation",
    "frame_interpolation",
    "optical_flow",
    "detection",
]


def python() -> Path:
    return HOME / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def model_root() -> Path:
    """A second folder, when selected, receives ComfyUI's models instead of Effigy's root."""
    return settings.comfyui_models().resolve()


def installed() -> bool:
    try:
        pin = json.loads((HOME / "installed.json").read_text(encoding="utf-8"))
        return (
            pin.get("revision") == REVISION
            and python().is_file()
            and (HOME / "source/main.py").is_file()
        )
    except (OSError, ValueError, AttributeError):
        return False


def installing() -> bool:
    return any(
        j.step == "install_comfyui" and j.status in ("queued", "running")
        for j in list(jobs.JOBS.values())
    )


def setup() -> dict:
    cfg = settings.read()
    return {
        "enabled": bool(cfg.get("comfyui_enabled", False)),
        "models": cfg.get("comfyui_models", ""),
        "download_path": str(model_root()),
        "installed": installed(),
        "installing": installing(),
        "generation_missing": generation_missing(),
    }


def generation_missing() -> dict[str, list[str]]:
    """The checkpoints Generate 3D lacks per generator (its Setup group's pinned files), in either model root."""
    roots = {model_root(), settings.models_root().resolve()}
    groups = {"fast": "comfy3d-fast", "faithful": "comfy3d", "faithful_mv": "comfy3d-mv"}
    return {
        key: [
            f
            for pinned, *_, filters in PINNED
            if group in pinned.split()
            for f in filters["allow_patterns"]
            if not any((r / f).is_file() for r in roots)
        ]
        for key, group in groups.items()
    }


def choose(enabled: bool, models: str) -> dict:
    if installing():
        raise ApiError(409, "busy", "wait for ComfyUI installation before changing its folders")
    path = models.strip()
    if path:
        new = Path(path)
        if "\x00" in path or not new.is_absolute() or new.is_file():
            raise ApiError(400, "bad_path", "choose a full models folder path or leave it blank")
        path = str(new.resolve())
    settings.write(comfyui_enabled=enabled, comfyui_models=path)
    return setup()


def prepare() -> None:
    """The selected root owns downloads; extra model paths are lookup-only."""
    (HOME / "data/custom_nodes").mkdir(parents=True, exist_ok=True)  # ComfyUI lists it at start
    shutil.copytree(
        ROOT / "comfy_nodes/effigy_3d", HOME / "data/custom_nodes/effigy_3d", dirs_exist_ok=True
    )
    model_root().mkdir(parents=True, exist_ok=True)
    paths = {k: k + "/" for k in MODEL_FOLDERS}
    paths.update(text_encoders="text_encoders/\nclip/", diffusion_models="diffusion_models/\nunet/")
    primary = settings.models_root().resolve()
    config = {"effigy": {"base_path": str(primary), **paths}} if primary != model_root() else {}
    (HOME / "extra_model_paths.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")


def environment() -> dict:
    root = model_root()
    return jobs._env() | {
        "HF_HUB_CACHE": str(root / "huggingface"),
        "HUGGINGFACE_HUB_CACHE": str(root / "huggingface"),
        "TORCH_HOME": str(root / "torch"),
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }


def command() -> list[str]:
    pin = json.loads((HOME / "installed.json").read_text(encoding="utf-8"))
    return [
        str(python()),
        str(HOME / "source/main.py"),
        "--listen",
        "127.0.0.1",
        "--base-directory",
        str(HOME / "data"),
        "--models-directory",
        str(model_root()),
        "--extra-model-paths-config",
        str(HOME / "extra_model_paths.yaml"),
        "--offline",
        "--disable-auto-launch",
        *(["--cpu"] if pin.get("device") == "cpu" else []),
    ]


# --- the running instance: Effigy's own, one at a time, on the GPU the steps use (jobs._env pins it) ---

_lock = threading.RLock()
_proc: subprocess.Popen | None = None
_port = 0
loaded: str | None = None  # the workflow whose models it may still hold (server/jobs)
free0 = 0  # bytes of video memory free on its GPU once it was up, before any model


def running() -> bool:
    return bool(_proc) and _proc.poll() is None


def url() -> str:
    return f"http://127.0.0.1:{_port}"


def request(path: str, body: dict | None = None, timeout: float = 10) -> dict:
    """GET `path` of the running instance, or POST `body` to it."""
    if not running():
        raise ApiError(409, "not_running", "ComfyUI is not running")
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url() + path, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except (OSError, ValueError) as e:
        raise ApiError(502, "comfyui", f"ComfyUI {path}: {e}") from e


def start(timeout: float = 180) -> dict:
    """Start it on a free port and wait until it answers; a no-op when it runs."""
    global _proc, _port, loaded, free0
    with _lock:
        if running():
            return {"running": True, "url": url()}
        if not (settings.read().get("comfyui_enabled") and installed()):
            raise ApiError(409, "missing_env", "Turn on and install ComfyUI in Setup first")
        stop()  # one a killed backend left behind
        prepare()
        with socket.socket() as s:  # a port nobody has: the user's own ComfyUI keeps 8188
            s.bind(("127.0.0.1", 0))
            _port = s.getsockname()[1]
        log = open(HOME / "comfyui.log", "w", encoding="utf-8")  # noqa: SIM115 - the child's
        env = environment()
        if gpu := jobs._smi("memory.total", "memory.free"):
            env["EFFIGY_COMFYUI_VRAM_GB"] = str(presets.budget_gb(gpu[0] / 1024, gpu[1] / 1024))
        _proc = subprocess.Popen(
            # a GB kept free on the card, other programs' share counted: a model that fits must never push
            # Windows into shared memory (steps/_comfy reports it if it happens)
            [*command(), "--port", str(_port), "--vram-headroom", "1"],
            env=env,
            cwd=HOME / "source",
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        mine = {"pid": _proc.pid, "port": _port, "owner": os.getpid()}
        (HOME / "running.json").write_text(json.dumps(mine), encoding="utf-8")
        loaded, end = None, time.time() + timeout
        while time.time() < end and running():
            try:
                free0 = request("/system_stats", timeout=2)["devices"][0]["vram_free"]
                return {"running": True, "url": url()}
            except ApiError:
                time.sleep(0.5)
        stop()
        raise ApiError(502, "comfyui", "ComfyUI did not start: see comfyui/comfyui.log")


def _alive(pid: int) -> bool:
    if os.name == "nt":
        cmd = ["tasklist", "/FI", f"PID eq {pid}", "/NH"]
        return str(pid) in subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        pass
    return True


def stop() -> dict:
    """Kill it and wait for its port to close: the video memory is back when the process is gone. Also the
    instance a killed backend left behind (comfyui/running.json), which would keep its models on the GPU."""
    global _proc, _port, loaded
    with _lock:
        proc, _proc, loaded = _proc, None, None
        if not proc:
            try:
                left = json.loads((HOME / "running.json").read_text(encoding="utf-8"))
                if not _alive(left["owner"]):  # another backend's (a test run beside the app) stays
                    _port = left["port"]
                    with urllib.request.urlopen(
                        url() + "/system_stats", timeout=2
                    ):  # still ComfyUI
                        proc = SimpleNamespace(pid=left["pid"], wait=lambda: None)
            except (OSError, ValueError, KeyError):
                pass
        if proc:
            jobs._kill_tree(proc)
            proc.wait()
            for _ in range(50):  # Windows: the launcher can exit just before its child lets go
                with socket.socket() as s:
                    if s.connect_ex(("127.0.0.1", _port)):
                        break
                time.sleep(0.2)
            (HOME / "running.json").unlink(missing_ok=True)
        return {"running": False}


def ready(name: str) -> None:
    """Running, and holding no other workflow's models than `name`'s.
    ponytail: another workflow means a restart (about 11 s): the one unload that cannot leave memory behind.
    `/free` and a check of the free memory (steps/_comfy.release) is the faster way if switching gets frequent."""
    global loaded
    with _lock:
        if running() and loaded not in (None, name):
            stop()
        start()
        loaded = name


def interrupt() -> None:
    """Drop what it runs and what waits: a cancelled job's prompt (the instance is Effigy's alone)."""
    if running():
        request("/queue", {"clear": True})
        request("/interrupt", {})


def workflows() -> list[dict]:
    """The image and edit workflows for the UI, each with the model files its graph names and the folders lack;
    none while ComfyUI is off or not installed."""
    from steps import _comfy

    if not (settings.read().get("comfyui_enabled") and installed()):
        return []
    roots = {model_root(), settings.models_root().resolve()}
    out = []
    for name, m in _comfy.workflows().items():
        graph = json.loads(Path(m["file"]).read_text(encoding="utf-8"))
        files = {
            v
            for node in graph.values()
            for v in node.get("inputs", {}).values()
            if isinstance(v, str) and v.endswith((".safetensors", ".gguf", ".ckpt", ".pt", ".sft"))
        }
        missing = sorted(f for f in files if not any(any(r.glob(f"*/{f}")) for r in roots))
        keys = ("title", "kind", "about", "restricted", "own", "models")
        out.append(
            {"id": name, "missing": missing, "reference": "reference" in m["set"]}
            | {k: m.get(k) for k in keys}
        )
    return out


def install() -> jobs.Job:
    from server import install as installer

    with installer._asking:
        if not settings.read().get("comfyui_enabled"):
            raise ApiError(409, "disabled", "enable ComfyUI before installing it")
        if installing():
            raise ApiError(409, "busy", "ComfyUI is already being installed")
        job = jobs.Job("", "install_comfyui", None, {})
        jobs.JOBS[job.id] = job
    threading.Thread(target=_install, args=(job,), daemon=True).start()
    return job


def _install(job: jobs.Job) -> None:
    from server import install as installer

    with installer._turn:
        with jobs._state:
            if job.status == "cancelled":
                return
            job.status = "running"
        jobs.publish("job", job.public())
        try:
            rc, lines = installer._cmd(
                job, [str(jobs.venv_python(".venv")), "-m", "server.comfyui_install"]
            )
            if rc or not installed():
                raise RuntimeError(lines[-1] if lines else "ComfyUI install did not finish")
            status, error = "done", None
        except Exception as e:  # noqa: BLE001 - installer failures must finish the visible job
            status, error = "failed", {"code": "failed", "message": str(e)}
        with jobs._state:
            if job.status != "cancelled":
                job.status, job.error = status, error
        jobs.publish("job", job.public())


if __name__ == "__main__":
    if not installed():
        raise SystemExit("Install ComfyUI in Setup first")
    prepare()
    raise SystemExit(
        subprocess.run(
            command() + sys.argv[1:], env=environment(), cwd=HOME / "source", check=False
        ).returncode
    )
