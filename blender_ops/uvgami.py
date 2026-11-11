# SPDX-License-Identifier: GPL-3.0-or-later
"""in: [mesh (OBJ keeps quads, or GLB)]; params: {"priority": "BALANCED"|"LESS_STRETCH"|"FEWER_SEAMS",
"symmetry": true, "timeout": 1800} → mesh.glb with UVs (UVgami's OptCuts unwrap, packed) + report.

UVgami (server/uvgami_install: its own profile in BLENDER_USER_RESOURCES, its engine in UVGAMI_ENGINE_DIR) is
enabled here: run_blender starts Blender with --factory-startup. Its unwrap is asynchronous, driven by
bpy.app.timers, which a background Blender never ticks: the timers are pumped by hand (UVgami's own tests do the
same, dev/tests/blender/timer_pump.py).
"""

import importlib
import sys
import time
from pathlib import Path

import addon_utils
import bmesh
import bpy

sys.path.insert(0, str(Path(__file__).parent))
from _lib import done, export_glb, import_mesh, job, mesh_report

MODULE = "bl_ext.user_default.uvgami"


class Pump:
    """bpy.app.timers' register/is_registered/unregister, ticked by run_until."""

    def __init__(self):
        self.pending = []

    def register(self, function, first_interval=0, persistent=False):
        self.pending.append((time.monotonic() + first_interval, function))

    def is_registered(self, function):
        return any(function is f for _, f in self.pending)

    def unregister(self, function):
        self.pending = [e for e in self.pending if e[1] is not function]

    def run_until(self, condition, timeout):
        deadline = time.monotonic() + timeout
        while not condition():
            if time.monotonic() > deadline:
                raise TimeoutError(f"UVgami still running after {timeout:.0f} s")
            now = time.monotonic()
            due = sorted((e for e in self.pending if e[0] <= now), key=lambda e: e[0])
            self.pending = [e for e in self.pending if e[0] > now]
            for _, function in due:
                interval = function()
                if interval is not None:
                    self.pending.append((time.monotonic() + interval, function))
            time.sleep(0.01)


j = job()
p = j["params"]
obj = import_mesh(j["in"][0])  # resets to factory settings: the add-on is enabled after it
# one vertex per position, as the cumesh path's weld (a flat-shaded GLB comes in face by face, each unwrapped
# alone), and smooth: flat or with its custom normals, the export splits every face's corners again
bm = bmesh.new()
bm.from_mesh(obj.data)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
bm.to_mesh(obj.data)
bm.free()
for poly in obj.data.polygons:
    poly.use_smooth = True
if obj.data.has_custom_normals:
    bpy.ops.mesh.customdata_custom_splitnormals_clear()
addon_utils.enable(MODULE, default_set=True)
addon = importlib.import_module(MODULE)
manager = importlib.import_module(MODULE + ".src.manager").manager
prefs = addon.src.utils.paths.get_preferences()
# off: the progress bar needs a GPU context, a popup crashes a background Blender
for name in ("autosave", "show_progress_bar", "show_popup"):
    setattr(prefs, name, False)

props = bpy.context.scene.uvgami
props.engine = "OPTCUTS"
props.priority = p.get("priority", "BALANCED")
props.symmetry = bool(p.get("symmetry", True))
props.keep_quads = True
props.pack_after_unwrap = True
props.use_proxy = "proxy_faces" in p
if props.use_proxy:
    props.proxy_faces = int(p["proxy_faces"])
t0 = time.time()
pump = Pump()
saved = {n: getattr(bpy.app.timers, n) for n in ("register", "is_registered", "unregister")}
for n in saved:
    setattr(bpy.app.timers, n, getattr(pump, n))
try:
    bpy.ops.uvgami.start()
    pump.run_until(lambda: manager.is_active, 120)
    pump.run_until(lambda: not manager.is_active, float(p.get("timeout", 1800)))
except Exception:
    manager.cancel_session()  # stop OptCuts too when a dense high-poly unwrap times out
    raise
finally:
    for n, f in saved.items():
        setattr(bpy.app.timers, n, f)
out = [o for o in bpy.data.objects if o.name.endswith("_unwrapped")]
errors = [str(e) for e in manager.error_messages]
assert out and out[0].data.uv_layers and not errors, f"UVgami failed: {manager.summary} {errors}"
bpy.ops.object.select_all(action="DESELECT")
out[0].select_set(True)
bpy.context.view_layer.objects.active = out[0]
# a plain material: without one the glTF exporter drops the UVs (export_materials NONE) or trimesh skips them on
# reading (steps/_mesh.read_glb: no material, no TextureVisuals)
if not out[0].data.materials:
    out[0].data.materials.append(bpy.data.materials.new("uvgami"))
export_glb(Path(j["out_dir"]) / "mesh.glb", use_selection=True)
rep = mesh_report(out[0])
done(
    j,
    {
        "outputs": {"mesh": "mesh.glb"},
        "report": rep
        | {
            "unwrapper": "uvgami",
            "engine": "OPTCUTS",
            "priority": props.priority,
            "seconds": round(time.time() - t0, 1),
        },
    },
)
