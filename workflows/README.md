# Image workflows

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
