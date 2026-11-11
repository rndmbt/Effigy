#!/usr/bin/env sh
# Start Effigy: sync the backend env, build the UI once, serve on 127.0.0.1 (port 8000 unless chosen in Setup or
# set in EFFIGY_PORT; python -m server opens the browser there).
set -e
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "uv not found: install it from https://docs.astral.sh/uv/"; exit 1; }
uv sync --frozen
UV_PROJECT_ENVIRONMENT=.venv-ml uv sync --frozen --group ml  # steps' venv; a no-op once synced
uv run python -m server.blender_install  # pinned Blender, first run only
uv run python -m server.gltfpack_install  # pinned gltfpack (LODs, web profile), first run only
# Optional backend failure must not prevent opening Setup to retry or disable it.
uv run python -m server.comfyui_install --if-enabled >> run.log 2>&1 || printf '%s\n' 'Optional ComfyUI setup failed. Open Setup to retry or disable it. Details in run.log.' >&2
[ -f ui/dist/index.html ] || (cd ui && npm ci && npm run build)
exec uv run python -m server
