"""Install ComfyUI's pinned source and isolated environment; no models are downloaded."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from server import comfyui, jobs, settings

SOURCE_URL = f"https://codeload.github.com/Comfy-Org/ComfyUI/zip/{comfyui.REVISION}"
SOURCE_SHA = "1fec46fc59ab42aaf06d507ce75b9c5bee8e7d17eb4c3bb35079295a6c386cda"
TORCH = "2.14.1"
VISION = "0.29.1"


def source() -> Path:
    home, tree = comfyui.HOME, comfyui.HOME / "source"
    if (
        (tree / ".effigy-revision").is_file()
        and (tree / ".effigy-revision").read_text().strip() == comfyui.REVISION
        and (tree / "main.py").is_file()
    ):
        return tree
    if tree.exists():
        raise RuntimeError(
            "comfyui/source is incomplete or from another revision; move it aside and retry"
        )
    home.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=home, prefix="install-") as temp:
        temp = Path(temp)
        archive = temp / "source.zip"
        req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Effigy installer"})
        with urllib.request.urlopen(req, timeout=120) as response, archive.open("wb") as f:
            shutil.copyfileobj(response, f)
        with archive.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != SOURCE_SHA:
                raise ValueError("ComfyUI source SHA-256 mismatch; not installing")
        with zipfile.ZipFile(archive) as z:
            z.extractall(temp)
        unpacked = temp / f"ComfyUI-{comfyui.REVISION}"
        (unpacked / ".effigy-revision").write_text(comfyui.REVISION, encoding="utf-8")
        unpacked.rename(tree)
    return tree


def install() -> None:
    if not comfyui.installed():
        tree = source()
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT")
        }
        if not comfyui.python().is_file():
            subprocess.run(
                [jobs.uv(), "venv", "--python", "3.12", str(comfyui.HOME / ".venv")],
                env=env,
                check=True,
            )
        smi = shutil.which("nvidia-smi")
        cuda = bool(
            smi
            and subprocess.run([smi, "-L"], capture_output=True, check=False, timeout=15).returncode
            == 0
        )
        device = "cu130" if cuda else "cpu"
        subprocess.run(
            [
                jobs.uv(),
                "pip",
                "install",
                "--python",
                str(comfyui.python()),
                "--index-url",
                "https://pypi.org/simple",
                "--extra-index-url",
                f"https://download.pytorch.org/whl/{device}",
                "--index-strategy",
                "unsafe-best-match",
                f"torch=={TORCH}+{device}",
                f"torchvision=={VISION}+{device}",
                "-r",
                str(tree / "requirements.txt"),
            ],
            env=env,
            check=True,
        )
        (comfyui.HOME / "installed.json").write_text(
            json.dumps(
                {
                    "revision": comfyui.REVISION,
                    "torch": TORCH,
                    "device": "cuda" if cuda else "cpu",
                }
            ),
            encoding="utf-8",
        )
    comfyui.prepare()
    print(f"ComfyUI installed: {comfyui.HOME}; models: {comfyui.model_root()}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--if-enabled", action="store_true", help="install only when selected in Effigy Setup"
    )
    args = parser.parse_args(argv)
    if not args.if_enabled or settings.read().get("comfyui_enabled"):
        install()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
