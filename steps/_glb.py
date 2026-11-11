"""A GLB's JSON and binary chunks, read and written back without a glTF library (stdlib + numpy).

For edits that keep the geometry: renaming nodes (parts rename, M31b), rewriting skin weights in place
(rigid parts, M32). The binary chunk keeps its layout, so accessors stay valid.
"""

import json
from pathlib import Path

import numpy as np

DTYPES = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read(path: Path) -> tuple[dict, bytearray]:
    data = Path(path).read_bytes()
    n = int.from_bytes(data[12:16], "little")
    rest = data[20 + n :]
    blob = rest[8 : 8 + int.from_bytes(rest[:4], "little")] if rest else b""
    return json.loads(data[20 : 20 + n]), bytearray(blob)


def write(path: Path, doc: dict, blob: bytearray) -> None:
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)  # chunks are 4-byte aligned, JSON padded with spaces
    chunks = len(js).to_bytes(4, "little") + b"JSON" + js
    if blob:
        b = bytes(blob) + b"\0" * (-len(blob) % 4)
        chunks += len(b).to_bytes(4, "little") + b"BIN\0" + b
    Path(path).write_bytes(
        b"glTF" + (2).to_bytes(4, "little") + (12 + len(chunks)).to_bytes(4, "little") + chunks
    )


def accessor(doc: dict, blob: bytearray, i: int) -> np.ndarray:
    """Accessor i as a writable (count, width) view into blob (no sparse accessors: ours have none)."""
    a = doc["accessors"][i]
    view = doc["bufferViews"][a["bufferView"]]
    dt, width = np.dtype(DTYPES[a["componentType"]]), WIDTH[a["type"]]
    return np.ndarray(
        (a["count"], width),
        dt,
        buffer=blob,
        offset=view.get("byteOffset", 0) + a.get("byteOffset", 0),
        strides=(view.get("byteStride") or width * dt.itemsize, dt.itemsize),
    )


def rotate(src: Path, dst: Path, R: np.ndarray, shift=(0.0, 0.0, 0.0), scale=1.0) -> str | None:
    """Write src to dst with its positions turned by `v @ R`, scaled by `scale` (a number, or one per axis after the
    turn) and moved by `shift`, its normals and tangents turned (R a pure turn; under a per-axis scale normals take
    its inverse and tangents the scale, both made unit length again; the layout, UVs, indices and materials
    untouched; POSITION min/max updated). None when done; else why not (dst not written): the file must be one
    mesh node alone in its scene (no parent transform), an identity transform, no skin or morph targets, plain
    float vertex data (a rigged or hand-placed mesh is not ours to turn)."""
    doc, blob = read(src)
    nodes = doc.get("nodes", [])
    meshes = [i for i, n in enumerate(nodes) if "mesh" in n]
    if len(meshes) != 1 or len(doc.get("meshes", [])) != 1:
        return "it is not a single mesh"
    if doc.get("skins") or "skin" in nodes[meshes[0]]:
        return "it has a skin"
    n = nodes[meshes[0]]
    if (doc.get("scenes") or [{}])[0].get("nodes") != meshes:
        return "its mesh sits under another node"
    ident = {
        "matrix": np.eye(4).ravel().tolist(),
        "rotation": [0, 0, 0, 1],
        "translation": [0, 0, 0],
        "scale": [1, 1, 1],
    }
    if any(n[k] != v for k, v in ident.items() if k in n):
        return "its mesh node has a transform"
    done = {}
    for p in doc["meshes"][0]["primitives"]:
        if "targets" in p:
            return "it has morph targets"
        for key in ("POSITION", "NORMAL", "TANGENT"):
            if key in p["attributes"]:
                a = doc["accessors"][p["attributes"][key]]
                if a["componentType"] != 5126 or "bufferView" not in a or "sparse" in a:
                    return "its vertex data is not plain float"
                done[p["attributes"][key]] = key
    sc = np.broadcast_to(np.asarray(scale, np.float32), 3)
    for i, key in done.items():
        a = accessor(doc, blob, i)
        a[:, :3] = a[:, :3] @ R.astype(np.float32)
        if key == "POSITION":
            a[:, :3] = a[:, :3] * sc + np.asarray(shift, np.float32)
            doc["accessors"][i]["min"], doc["accessors"][i]["max"] = (
                a.min(0).tolist(),
                a.max(0).tolist(),
            )
        elif not np.allclose(sc, sc[0]):  # a per-axis scale bends the normals and tangents
            a[:, :3] = a[:, :3] / sc if key == "NORMAL" else a[:, :3] * sc
            a[:, :3] /= np.maximum(np.linalg.norm(a[:, :3], axis=1, keepdims=True), 1e-12)
    write(dst, doc, blob)


def _key(doc: dict, blob: bytearray, att: dict) -> np.ndarray:
    """Position, UV and normal / 100 per vertex: Blender's normals (up to 0.4° off) move it under 7e-5 (copy_tangents
    takes up to 5e-4), the other side of a hard edge (normals tens of degrees apart) about 1e-2.
    """
    p, uv, n = (
        accessor(doc, blob, att[k]).astype(np.float64) for k in ("POSITION", "TEXCOORD_0", "NORMAL")
    )
    return np.hstack([p, uv, n / 100])


def copy_tangents(src: Path, dst: Path) -> int:
    """`src`'s TANGENT written into each primitive of `dst` that has none, vertex by vertex matched on position,
    UV and normal (a hard edge's two copies differ only in it; Blender writes normals a few 0.1° off) (Blender's
    skinning and clip exports drop it; without it a viewer builds its own frame and the baked normal map shades
    differently). A primitive with a vertex that has no match is left alone.
    Returns how many primitives got one."""
    from scipy.spatial import cKDTree

    sd, sb = read(src)
    keys, tangents = [], []
    for mesh in sd.get("meshes", []):
        for prim in mesh["primitives"]:
            att = prim["attributes"]
            if {"POSITION", "TEXCOORD_0", "NORMAL", "TANGENT"} <= set(att):
                keys.append(_key(sd, sb, att))
                tangents.append(accessor(sd, sb, att["TANGENT"]).astype(np.float32))
    if not keys:
        return 0
    tree, tangents = cKDTree(np.concatenate(keys)), np.concatenate(tangents)
    doc, blob = read(dst)
    done = 0
    for mesh in doc.get("meshes", []):
        for prim in mesh["primitives"]:
            att = prim["attributes"]
            if "TANGENT" in att or not {"POSITION", "TEXCOORD_0", "NORMAL"} <= set(att):
                continue
            gap, at = tree.query(_key(doc, blob, att))
            if len(gap) and gap.max() > 5e-4:
                continue
            data = np.ascontiguousarray(tangents[at], "<f4").tobytes()
            blob += b"\0" * (-len(blob) % 4)
            doc["bufferViews"].append(
                {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": 34962}
            )
            doc["accessors"].append(
                {
                    "bufferView": len(doc["bufferViews"]) - 1,
                    "componentType": 5126,
                    "count": len(at),
                    "type": "VEC4",
                }
            )
            att["TANGENT"] = len(doc["accessors"]) - 1
            blob += data
            done += 1
    if done:
        doc["buffers"][0]["byteLength"] = len(blob)
        write(dst, doc, blob)
    return done
