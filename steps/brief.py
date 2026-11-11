"""Prompt → brief.yaml from a category template (IMPLEMENTATION.md M10; no LLM yet).

params: {"prompt": "dwarf blacksmith", "category": "character"|"prop"|"creature"|"posed", "framing"?: the words
added after the prompt instead of the category's template ("" adds nothing)}
Templates encode what a "3D-ready" image is for TRELLIS.2 (research 07 §3). The 2D track's categories (research
15, M50: "picture" (nothing added), "wallpaper", "background", "portrait", "icon", "item") frame a picture for `concepts` op `image`; "tile"
(M54) a texture for `concepts` op `tile`.
"""

import re
from pathlib import Path

import yaml

from steps._common import StepError, main

# a prompt that sets its own view or camera; whole words ("an overview map", "a security camera", "a sign above a door"
# keep the template's)
OWN_VIEW = re.compile(
    r"\b(view|elevation|from above|from below|seen from|the camera|camera angle)\b"
)

TEMPLATES = {
    "character": "full body, symmetric neutral A-pose, feet side by side, straight arms about 45° from the "
    "body with a visible gap, empty open hands, nothing over the shoulders, front view, soft key light, "
    "plain light-grey background, no cast shadow, single character",
    # "the camera level at mid height" (the owner's agent, 2026-10-13: a concept shot from above makes a model that
    # leans 10-15°; with it the lamp's four concepts came out upright and the wall level)
    "prop": "single complete object centred, front three-quarter view, the camera level at mid height, even soft "
    "studio light, flat plain light-grey background, no cast shadow, product photograph",
    "creature": "full body, three-quarter side view, the camera level at mid height, all legs visible and apart, "
    "standing pose, soft key light, plain light-grey background, no cast shadow, single creature",
    # a character keeping the prompt's pose (M16 bake-off inputs, statues): framing without pose rules
    "posed": "full body, front view, soft key light, plain light-grey background, no cast shadow, "
    "single character",
    # the 2D track (research 15 §2, M50): pictures that stay pictures; "picture" adds nothing (the owner, 2026-10-13:
    # a picture made from a prompt as it is, what Z-Image does)
    "picture": "",
    "wallpaper": "wide scenic composition, rich detail, atmospheric light, no text, no watermark",
    "background": "side-scrolling game background, eye-level side view, an open floor band along the bottom, "
    "no characters, no text",
    "portrait": "head and shoulders portrait facing the viewer, plain background, game character portrait, "
    "no text",
    "icon": "single game icon centred, bold readable shape, plain flat background, no text",
    "item": "single game item centred, side view, the whole object in frame, plain flat light-grey "
    "background, no cast shadow",
    # M54: a texture that repeats; the material straight on and edge to edge (the seams are mended after)
    "tile": "seamless tileable texture, a flat lay seen straight from above, uniform scale everywhere with no "
    "perspective, large simple shapes with only a few across the frame, the material fills the frame edge to "
    "edge, even soft light, no shadows, no objects, no text",
}


# a size the prompt gives ("real-world size approximately 3.2 m long, 0.3 m deep, 2.1 m tall"): validate holds the model
# against it (the owner's agent, 2026-10-13: nothing said a hoarding came out 4.14 m long for a 3.2 m brief)
SIZE = re.compile(
    r"(?<![\d.,])(\d+(?:\.\d+)?|\.\d+)\s*(cm|metres?|meters?|m)[\s-]+(wide|long|deep|tall|high)\b"
)


def dims(prompt: str) -> dict:
    """{"height": m, "plan": [m, ...], "long": m} from the prompt's "N m tall / high / wide / long / deep" (cm, metres
    too; "2 m-tall"), the plan's largest first, "long" the one it calls long; {} when it gives none. ponytail: a
    reading, not a parser: "1,8 m" and a size in words are missed, a range's upper end is taken, and "a 3 m tall tree
    with 0.5 m long branches" gives the branches as its length."""
    out: dict = {}
    for n, unit, word in SIZE.findall(prompt.lower()):
        m = float(n) / (100 if unit == "cm" else 1)
        if m <= 0:
            continue
        if word in ("tall", "high"):
            out.setdefault("height", m)
            continue
        out["plan"] = sorted([*out.get("plan", []), m], reverse=True)
        if word == "long":
            out.setdefault("long", m)
    return out


def run(op, in_dir, out_dir, params):
    prompt, category = params.get("prompt", "").strip(), params.get("category", "prop")
    if not prompt or category not in TEMPLATES:
        raise StepError("bad_params", f"need a prompt and a category in {sorted(TEMPLATES)}")
    template = TEMPLATES[category]
    if isinstance(params.get("framing"), str):  # the user's own words for it, as they are
        template = params["framing"].strip()
    elif OWN_VIEW.search(prompt.lower()):  # its own view or camera: the template's go
        template = ", ".join(
            p for p in template.split(", ") if "view" not in p and "camera" not in p
        )
    brief = {
        "prompt": prompt,
        "category": category,
        "prompt_full": ", ".join(x for x in (prompt, template) if x),
    }
    (Path(out_dir) / "brief.yaml").write_text(
        yaml.safe_dump(brief, sort_keys=False), encoding="utf-8"
    )
    return {"outputs": {"brief": "brief.yaml"}}


if __name__ == "__main__":
    main(run)
