"""Models & Licences page (IMPLEMENTATION.md M14): pinned weights on disk, what is complete, remove one.

Local folders live under EFFIGY_MODELS, HF-cache repos under EFFIGY_MODELS/hub (the HF_HUB_CACHE the backend gives
its children). A row is installed when every file tools/models_manifest.json lists for it is there at its
size (M35b); installing and repairing: server/install.py.
"""

import json
import os
import shutil
from fnmatch import fnmatch
from pathlib import Path

from server import jobs, settings
from server.library import ApiError
from tools.download_models import FILES, MANIFEST, PINNED, key

MANIFEST_ROWS: dict[str, dict[str, list]] = json.loads(MANIFEST.read_text(encoding="utf-8"))

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {g for e in PINNED for g in e[0].split()} | {e[0] for e in FILES}


def root() -> Path:
    return settings.models_root()


def entries() -> list[dict]:
    """Every PINNED/FILES row; `where` is relative to EFFIGY_MODELS, `filters` only for local folders."""
    out = [
        {
            "name": repo,
            "groups": groups.split(),
            "revision": rev,
            "where": folder or f"hub/models--{repo.replace('/', '--')}",
            "filters": (filters or {}) if folder else {},  # HF-cache blobs have no names to filter
            "hub": not folder,
            "key": key(groups, repo, rev, folder),
        }
        for groups, repo, rev, folder, filters in PINNED
    ]
    # the licenses.yaml row name, e.g. https://github.com/valeoai/NAF/... → valeoai/NAF
    out += [
        {
            "name": "/".join(url.split("/")[3:5]),
            "groups": [group],
            "revision": f"sha256:{sha}",
            "where": path,
            "filters": {},
            "hub": False,
            "key": f"file:{path}",
        }
        for group, url, sha, path in FILES
    ]
    return out


def _files(e: dict) -> list[Path]:
    """The files of one entry (symlinks skipped: HF-cache snapshots point at blobs)."""
    base = root() / e["where"]
    if base.is_file():
        return [base]
    allow, ignore = e["filters"].get("allow_patterns"), e["filters"].get("ignore_patterns", [])
    out = []
    for d, dirs, names in os.walk(base):
        dirs[:] = [
            x for x in dirs if not os.path.isjunction(os.path.join(d, x))
        ]  # never through a link
        for n in names:
            f = Path(d, n)
            rel = f.relative_to(base).as_posix()
            if f.is_symlink() or (allow and not any(fnmatch(rel, p) for p in allow)):
                continue
            if not any(fnmatch(rel, p) for p in ignore):
                out.append(f)
    return out


def _size(f: Path) -> int:
    try:
        return f.stat().st_size
    except OSError:  # removed while listing
        return 0


def target(e: dict, rel: str) -> Path:
    """Where a manifest file of an entry is on disk (an HF-cache repo: in its pinned snapshot)."""
    base = root() / e["where"]
    if e["hub"]:
        return base / "snapshots" / e["revision"] / rel
    return base / rel if rel else base


def state(e: dict) -> dict:
    """want: the bytes the manifest lists for an entry; have: those on disk at their size; bad: the files
    missing or of another size (a symlink is followed to its blob)."""
    rows = MANIFEST_ROWS.get(e["key"], {})
    have, bad = 0, []
    for rel, (size, _) in rows.items():
        try:
            ok = os.stat(target(e, rel)).st_size == size
        except OSError:
            ok = False
        if ok:
            have += size
        else:
            bad.append(rel)
    return {"want": sum(size for size, _ in rows.values()), "have": have, "bad": bad}


def listing() -> list[dict]:
    out = []
    for i, e in enumerate(entries()):
        files, st = _files(e), state(e)
        info = {k: e[k] for k in ("name", "groups", "revision", "where")}
        installed = not st["bad"] if e["key"] in MANIFEST_ROWS else bool(files)
        out.append(
            {
                "id": i,
                **info,
                "installed": installed,
                "partial": bool(files) and not installed,  # some files there: Repair finishes it
                "bytes": sum(_size(f) for f in files),
                "missing_bytes": st["want"] - st["have"],
            }
        )
    return out


def _downloading() -> bool:
    """An install or a move of the models folder (server/install.py) is queued or running: files are
    changing."""
    return any(
        j.step in ("install", "install_comfyui", "move_models")
        and j.status in ("queued", "running")
        for j in list(jobs.JOBS.values())
    )


def remove(model_id: int, locked: bool = False) -> None:
    """Delete one known entry's files (ids index entries(); nothing from the request becomes a path).
    locked: the caller holds the GPU lock and has stopped the resident worker (server/storage.py)."""
    if not 0 <= model_id < len(es := entries()):
        raise ApiError(404, "not_found", f"no model {model_id}")
    e = es[model_id]
    if not locked and _downloading():
        raise ApiError(409, "busy", "a model download is running; remove after it ends")
    if not locked and not jobs.GPU.acquire(blocking=False):
        raise ApiError(409, "busy", "a step is running; remove models after it ends")
    try:
        if not locked:
            jobs.evict()  # the resident worker may hold these files open (Windows can't delete them)
        base = root() / e["where"]
        if any(os.path.isjunction(p) for p in (base, *base.parents) if root() in p.parents):
            raise ApiError(409, "linked", f"{e['where']} lies behind a link; remove it by hand")
        if e["filters"]:  # only this entry's files: Pixal3D's folder holds single- and multi-view
            for f in _files(e):
                f.unlink()
        elif base.is_dir():
            shutil.rmtree(base)
        else:
            base.unlink(missing_ok=True)
    except OSError as err:
        raise ApiError(409, "busy", f"could not delete {e['where']}: {err}") from err
    finally:
        if not locked:
            jobs.GPU.release()


def missing_bytes(group: str) -> int:
    """What installing a group still has to download: the manifest's files not on disk at their size."""
    return sum((st := state(e))["want"] - st["have"] for e in entries() if group in e["groups"])
