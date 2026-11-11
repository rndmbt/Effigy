# Workflows

## Generate 3D

`pixal3d_generate.json` is the generation graph, not an image-model drop-in. With ComfyUI selected in Setup,
Effigy binds it to Generate 3D. It uses native camera/shape/material/reconstruction nodes and the small
GPL-licensed boundary under `comfy_nodes/effigy_3d`, copied into the managed ComfyUI at startup. The existing
UVgami and bake steps finish the initial high-poly output without importing ComfyUI into the main environment.
Setup groups `comfy3d`, `comfy3d-fast`, `comfy3d-mv` install the pinned checkpoints. Uncheck ComfyUI to use the
original native generation path. Missing files never trigger a silent fallback.

Based on the [native Pixal3D template](https://docs.comfy.org/tutorials/3d/pixal3d). Repair settings are not
universal: solid reconstruction can close intended openings, and excessive smoothing loses fine detail.
Inspect the reference alignment, silhouette, openings and material channels, not just the topology report.

## Image workflows

Each pair of files here is a ComfyUI workflow that Effigy offers as an image model: `<id>.json` is the graph,
`<id>.yaml` says what Effigy sets in it and which models it loads. Concept images and pictures list the
`image` ones under "Image model", Edit picture the `edit` ones.

## Add your own

Put your pair in `comfyui/workflows/` (next to the ComfyUI install; that folder is yours and never
overwritten). The same id there replaces a shipped one.

1. Build the workflow in ComfyUI and run it once. It needs one Save Image or Preview Image node.
2. Export it with **File → Export (API)** and save it as `comfyui/workflows/my_model.json`.
3. Write `comfyui/workflows/my_model.yaml` beside it:

```yaml
title: My model
kind: image            # image: a description becomes a picture; edit: a picture and an instruction become a picture
about: One line shown under the model list.
set:                   # which node input gets which value: "<node id>.<input name>", as in the .json
  prompt: 6.text
  seed: 3.seed
  width: 5.width       # image only
  height: 5.height
  image: 10.image      # edit only: the Load Image node that receives the picture
models:                # what the graph loads
  - {name: author/model, license: Apache-2.0, commercial: true}
```

`prompt` is required, and `image` for an edit. A value you leave out of `set` stays as the graph has it.
A list (`seed: [3.seed, 9.noise_seed]`) sets several inputs.

## Licences

A result is marked **restricted** unless every model under `models` says `commercial: true`; a workflow
without a `models` list is restricted too. Restricted results work everywhere in Effigy. The Licences panel
names the model, and an export asks you to confirm non-commercial use and writes the note into `CREDITS.txt`.

## Video memory

Effigy starts its own ComfyUI for these workflows and stops it before any other step loads a model, so one
model is on the card at a time. A result whose run pushed video memory into shared memory says so in its note.

## Texture references

A texture workflow can bind `reference` to a second Load Image node. Texture detail supplies the original
prepared concept from the version chain; the `image` input remains the current rendered view. The supplied
Qwen texture workflow uses this path. Such workflows are offered in texture tools, not ordinary picture edits.
