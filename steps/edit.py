"""Change a picture with words: an edit workflow in ComfyUI redraws it (steps/_comfy.py).

In: any version with a picture: concept candidates or pictures (params `key`: which one, an output name such as
"cand_2" or "image_1"; default the first), an upload, a re-posed or cut-out reference, an earlier edit. params:
{"prompt": "give him a horned helmet, keep the rest", "workflow": the id of an edit workflow, "key"?, "seed"?}. A picture with transparency goes in on light grey.
Out: image.png, at the size the workflow gives it (the source's, about 1 MP). The parent chain stays, so the
edited picture goes on to `prep` like the one it came from.
"""

import json
import random
import time
from pathlib import Path

from PIL import Image

from steps import _comfy
from steps._common import StepError, main

GREY = (230, 230, 230)  # the references' plain background


def run(op, in_dir, out_dir, params):
    prompt, seed = str(params.get("prompt") or "").strip(), params.get("seed")
    seed = random.randrange(2**31) if seed is None else seed
    if not prompt or not params.get("workflow") or not isinstance(seed, int):
        raise StepError(
            "bad_params", "edit: a prompt (what to change), a workflow and a whole-number seed"
        )
    if not in_dir:
        raise StepError("missing_input", "edit needs a parent version with a picture")
    meta = json.loads((Path(in_dir) / "meta.json").read_text(encoding="utf-8"))
    outs = meta["outputs"]
    key = str(params.get("key") or "")
    src = (
        outs.get(key)
        if key.startswith(("cand_", "image"))
        else outs.get("image") or outs.get("cand_0")
    )
    if not src:
        raise StepError("missing_input", f"{meta['version']}: no picture to edit")
    t0, picture = time.time(), Image.open(Path(in_dir) / src)
    if picture.mode != "RGB":  # a cut-out: the edit models take no alpha
        rgba = picture.convert("RGBA")
        picture = Image.new("RGB", rgba.size, GREY)
        picture.paste(rgba, mask=rgba.getchannel("A"))
    images, result = _comfy.generate(params["workflow"], "edit", prompt, [seed], image=picture)
    images[0].convert("RGB").save(Path(out_dir) / "image.png")
    result["metrics"] |= {"seconds": round(time.time() - t0, 1), "seed": seed}
    return result | {"outputs": {"image": "image.png"}}


if __name__ == "__main__":
    main(run)
