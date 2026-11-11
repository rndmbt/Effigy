"""ComfyUI workflows as image models: text → picture ("image") and picture + instruction → picture ("edit").

A workflow is two files with one name: `<id>.json`, the graph as ComfyUI saves it with "Export (API)", and
`<id>.yaml`, what Effigy needs to know about it:

    title: Qwen-Image 2.1 Turbo
    kind: image                      # or edit
    about: one line shown under the model list
    rank: 10                         # optional: lower comes first in the list (default 50); the first edit
                                     # workflow is the one the UI starts with
    set:                             # an Effigy value → "<node id>.<input>" (or a list of them)
      prompt: 5.prompt
      seed: 10.noise_seed
      width: 6.width                 # image only
      height: 6.height
      image: 20.image                # edit only: the LoadImage that gets the picture
    models:                          # what the graph loads; anything not `commercial: true` (or no list at
      - {name: Qwen/Qwen-Image-2.1, license: qwen-research, commercial: false}   # all) marks the result restricted

`workflows/` holds the ones Effigy ships, `comfyui/workflows/` the user's own (the same id there wins). The
server (server/jobs) starts ComfyUI, hands its address over in EFFIGY_COMFYUI and stops it before any other
GPU step; this module only talks HTTP to it (stdlib), so ComfyUI's GPL code stays in its own process.
"""

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import yaml
from PIL import Image

from steps._common import StepError, progress

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = [ROOT / "workflows", ROOT / "comfyui" / "workflows"]  # shipped, then the user's own
KINDS = ("image", "edit")
SPILL_GB = 0.3  # shared video memory a run may add before its result says so (the desktop's own moves less)


def workflows() -> dict[str, dict]:
    """{id: manifest} of every workflow with both files and a known kind; `restricted` and `own` added."""
    out = {}
    for own, folder in enumerate(FOLDERS):
        for file in sorted(folder.glob("*.yaml")):
            try:
                m = yaml.safe_load(file.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError):
                continue
            if not isinstance(m, dict) or m.get("kind") not in KINDS:
                continue
            if not file.with_suffix(".json").is_file() or not isinstance(m.get("set"), dict):
                continue
            models = [x for x in m.get("models") or [] if isinstance(x, dict) and x.get("name")]
            out[file.stem] = m | {
                "title": str(m.get("title") or file.stem),
                "models": models,
                "restricted": not models or not all(x.get("commercial") is True for x in models),
                "own": bool(own),
                "file": str(file.with_suffix(".json")),
            }
    return dict(sorted(out.items(), key=lambda kv: (kv[1].get("rank", 50), kv[1]["title"])))


def rows() -> list[dict]:
    """licenses.yaml-shaped rows of the workflows' models: the Licences panel and an export's credits name them."""
    seen = {}
    for m in workflows().values():
        for x in m["models"]:
            seen.setdefault(
                x["name"],
                {
                    "name": x["name"],
                    "kind": "weights",
                    "license": str(x.get("license") or "not stated"),
                    "commercial": x.get("commercial") is True,
                    "attribution": str(x.get("attribution") or ""),
                    "flag": "a ComfyUI workflow's model",
                },
            )
    return list(seen.values())


def manifest(name: str, kind: str) -> dict:
    m = workflows().get(str(name))
    if not m or m["kind"] != kind:
        have = sorted(k for k, v in workflows().items() if v["kind"] == kind)
        raise StepError("bad_params", f"workflow {name!r}: not one of the {kind} workflows {have}")
    return m


def provenance(m: dict) -> dict:
    """The RESULT fields a workflow adds: its models, their licences, restricted and the note that says why."""
    out = {
        "models": [{"id": x["name"], "license": str(x.get("license", ""))} for x in m["models"]],
        "licenses": [x["name"] for x in m["models"]],
    }
    if m["restricted"]:
        bad = [
            f"{x['name']} ({x.get('license') or 'licence not stated'})"
            for x in m["models"]
            if x.get("commercial") is not True
        ]
        why = ", ".join(bad) or "a workflow that does not list its models"
        out |= {"restricted": True, "note": f"Not cleared for commercial use: {why}."}
    return out


# --- HTTP ---------------------------------------------------------------------------------------


def _url() -> str:
    if not (url := os.environ.get("EFFIGY_COMFYUI")):
        raise StepError(
            "failed", "ComfyUI is not running (EFFIGY_COMFYUI): a workflow runs through the app"
        )
    return url


def _call(path: str, body: dict | None = None, raw: bool = False):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(_url() + path, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            got = r.read()
    except urllib.error.HTTPError as e:
        got = e.read()
        try:
            return json.loads(got)  # /prompt's 400 says which node refused which value
        except ValueError:
            raise StepError("failed", f"ComfyUI {path}: {e}") from e
    except OSError as e:
        raise StepError("failed", f"ComfyUI {path}: {e}") from e
    return got if raw else json.loads(got or b"{}")


def _upload(image: Image.Image) -> str:
    """The picture into ComfyUI's input folder; the name a LoadImage takes."""
    buf, mark = BytesIO(), uuid.uuid4().hex
    name = f"effigy_{mark}.png"
    image.save(buf, "PNG")
    body = (
        (
            f'--{mark}\r\nContent-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
            f'--{mark}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
            "Content-Type: image/png\r\n\r\n"
        ).encode()
        + buf.getvalue()
        + f"\r\n--{mark}--\r\n".encode()
    )
    req = urllib.request.Request(
        _url() + "/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={mark}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["name"]
    except (OSError, ValueError, KeyError) as e:
        raise StepError("failed", f"ComfyUI upload: {e}") from e


def _refused(reply: dict) -> str:
    """ComfyUI's validation error in a sentence: mostly a model file the graph names and the folder lacks."""
    lines = [
        f"{node.get('class_type')}: {e.get('message')} ({e.get('details')})"
        for node in (reply.get("node_errors") or {}).values()
        for e in node.get("errors", [])
    ]
    head = (reply.get("error") or {}).get("message", "the workflow was refused")
    return f"ComfyUI: {head}. " + "; ".join(lines)[:1500]


def execute(graph: dict) -> dict:
    """Run a graph to its end; return its output records (images or geometry files)."""
    reply = _call("/prompt", {"prompt": graph, "client_id": "effigy"})
    if not (pid := reply.get("prompt_id")):
        raise StepError("failed", _refused(reply))
    while not (done := _call(f"/history/{pid}").get(pid)):
        time.sleep(0.5)
    status = done.get("status", {})
    if status.get("status_str") != "success":
        errors = [m[1] for m in status.get("messages", []) if m[0] == "execution_error"]
        text = errors[0].get("exception_message", "") if errors else "interrupted"
        kind = errors[0].get("exception_type", "") if errors else ""
        oom = "OutOfMemory" in kind or "out of memory" in text.lower()
        raise StepError("oom" if oom else "failed", f"ComfyUI: {kind} {text}".strip()[:1500])
    return done.get("outputs", {})


def _queue(graph: dict) -> list[Image.Image]:
    """Run one image graph; fetch its output pictures."""
    images = []
    for node in execute(graph).values():
        for im in node.get("images", []):
            raw = _call("/view?" + urllib.parse.urlencode(im), raw=True)
            images.append(Image.open(BytesIO(raw)).copy())
    if not images:
        raise StepError(
            "failed", "ComfyUI: the workflow saved no image (it needs a Save or Preview Image node)"
        )
    return images


def _shared_gb() -> float | None:
    """Shared video memory (GB) in use on the busiest adapter; None off Windows. A model that fits must not
    add to it: Windows pages VRAM into system RAM there and every step runs many times slower."""
    if os.name != "nt":
        return None
    try:
        counters = [rf"\GPU Adapter Memory(*)\{k} Usage" for k in ("Dedicated", "Shared")]
        out = subprocess.run(
            ["typeperf", *counters, "-sc", "1"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
        row = [float(v.strip('"') or 0) for v in out.splitlines()[2].split(",")[1:]]
        half = len(row) // 2
        busiest = max(range(half), key=lambda i: row[i])
        return row[half + busiest] / 2**30
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


@contextmanager
def spill_watch():
    """Watch shared video memory while the block runs; yields a function giving the GB the block has added
    (None where it cannot be read)."""
    before, peak, stop = _shared_gb(), [0.0], threading.Event()

    def watch():
        while not stop.wait(2):
            if (now := _shared_gb()) is not None:
                peak[0] = max(peak[0], now)

    if before is not None:
        threading.Thread(target=watch, daemon=True).start()
    try:
        yield lambda: None if before is None else round(max(0.0, peak[0] - before), 2)
    finally:
        stop.set()


def generate(
    name: str,
    kind: str,
    prompt: str | list[str],
    seeds: list[int],
    width: int | None = None,
    height: int | None = None,
    image: Image.Image | None = None,
    p0: float = 0.05,
    p1: float = 0.9,
    reference: Image.Image | None = None,
) -> tuple[list[Image.Image], dict]:
    """One picture per seed (and per prompt, when a list) from workflow `name` (the first image each run saves), and its RESULT fields
    (`provenance` + metrics `shared_gb`: the most shared video memory the runs added)."""
    m = manifest(name, kind)
    graph = json.loads(Path(m["file"]).read_text(encoding="utf-8"))
    prompts = [prompt] * len(seeds) if isinstance(prompt, str) else prompt
    values = {"width": width, "height": height}
    if image is not None:
        values["image"] = _upload(image)
    if reference is not None and "reference" in m["set"]:
        values["reference"] = _upload(reference)
    if missing := [k for k in ["prompt", "image"][: 1 + (kind == "edit")] if k not in m["set"]]:
        raise StepError("bad_params", f"workflow {name}: its `set` needs {missing}")
    out = []
    with spill_watch() as added:
        for k, seed in enumerate(seeds):
            progress(p0 + (p1 - p0) * k / len(seeds), f"{m['title']}: picture {k + 1}/{len(seeds)}")
            for key, value in (values | {"seed": seed, "prompt": prompts[k]}).items():
                targets = m["set"].get(key)
                for target in [targets] if isinstance(targets, str) else targets or []:
                    node, _, field = str(target).partition(".")
                    if node not in graph or not field:
                        raise StepError(
                            "bad_params", f"workflow {name}: set.{key} names {target!r}"
                        )
                    if value is not None:
                        graph[node]["inputs"][field] = value
            out.append(_queue(graph)[0])
    result = provenance(m)
    spill = added()
    result["metrics"] = {"workflow": name, "shared_gb": spill}
    if spill is not None and spill > SPILL_GB:
        result["note"] = (
            result.get("note", "")
            + f" Video memory spilled: {spill:.1f} GB went to shared memory, so this ran slower than it should."
        ).strip()
    return out, result


def release(tolerance_gb: float = 0.7, seconds: float = 30) -> None:
    """Have ComfyUI unload its models and wait until the video memory is back: before this step puts a
    model of its own on the GPU. `/free` only sets a flag, so the free memory is read until it is back to
    what ComfyUI started with (EFFIGY_COMFYUI_FREE, bytes); a ComfyUI that keeps it fails the step."""
    base = float(os.environ.get("EFFIGY_COMFYUI_FREE") or 0)
    _call("/free", {"unload_models": True, "free_memory": True})
    end = time.time() + seconds
    while True:
        free = _call("/system_stats")["devices"][0]["vram_free"]
        if not base or free >= base - tolerance_gb * 2**30:
            return
        if time.time() > end:
            raise StepError(
                "failed",
                f"ComfyUI kept {(base - free) / 2**30:.1f} GB of video memory after unloading its models",
            )
        time.sleep(0.5)
