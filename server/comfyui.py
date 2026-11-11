"""Optional ComfyUI installation and model paths; ComfyUI runs in its own environment."""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import yaml

from server import jobs, settings
from server.library import ApiError

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
    return Path(settings.read().get("comfyui_models") or settings.models_root()).resolve()


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
    HOME.mkdir(parents=True, exist_ok=True)
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
