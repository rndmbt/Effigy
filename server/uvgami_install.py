"""Pinned UVgami (DanielBoxer/UVgami, GPL-3.0; 2026-11-04): the `uvgami` unwrapper of steps/uv, a Blender add-on
run in Effigy's Blender (blender_ops/uvgami.py) with its own user profile, and its OptCuts engine (MIT; the
engine's licences ship in its folder). Opt-in: installed once, SHA-256 checked (`EFFIGY_UVGAMI` overrides the
folder):

    uv run python -m server.uvgami_install

Only OptCuts: UVgami's PartUV engine segments with an NVIDIA model that is non-commercial (Effigy's licence
policy). The add-on is unpacked as an extension of the profile's user_default repository; nothing is installed
into Effigy's own Blender profile.
"""

import os
import platform
from pathlib import Path

from server.blender_install import ROOT, fetch

HOME = Path(os.environ.get("EFFIGY_UVGAMI") or ROOT / "uvgami")
REL = "https://github.com/DanielBoxer/UVgami/releases/download"
ADDON = (
    f"{REL}/v2.3.0/UVgami.zip",
    "38976ba74c6e7b6dca89ec3d81684a3352bea8d8bd117004ef99d483ec7dc91d",
)
# hashed by us when pinning (the releases list none).
# ponytail: Windows only; pin the Linux and macOS engines when someone runs there
ENGINE = {
    "Windows": (
        f"{REL}/optcuts-v1.23.1/optcuts-engine-1.23.1-windows.zip",
        "c545846fd2c711cd9cef90b3e9940d7acc60d3f34f258c4fb30cb434c975d4bd",
        "optcuts.exe",
    ),
}


def profile() -> Path:
    """The Blender user profile (BLENDER_USER_RESOURCES) UVgami lives in."""
    return HOME / "profile"


def addon_dir() -> Path:
    return profile() / "extensions" / "user_default" / "uvgami"


def engine_dir() -> Path:
    """UVGAMI_ENGINE_DIR: the folder holding the engine binary."""
    return HOME / "engine" / "bin"


def installed() -> bool:
    e = ENGINE.get(platform.system())
    return (
        (addon_dir() / "blender_manifest.toml").is_file()
        and bool(e)
        and (engine_dir() / e[2]).is_file()
    )


def install() -> Path:
    if platform.system() not in ENGINE:
        raise SystemExit("UVgami's pinned OptCuts engine is Windows only here")
    if not (addon_dir() / "blender_manifest.toml").is_file():
        fetch(*ADDON, addon_dir())  # the zip holds the extension's files at its top level
    url, sha, exe = ENGINE[platform.system()]
    if not (engine_dir() / exe).is_file():
        fetch(url, sha, HOME / "engine")
    return HOME


if __name__ == "__main__":
    print(install())
