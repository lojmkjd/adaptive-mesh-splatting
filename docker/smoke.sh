#!/usr/bin/env bash
#
# Smoke proof for the container: import the compiled stack, then run a short
# real training pass against a bind-mounted COLMAP scene and assert that a
# checkpoint was written under /outputs.
#
#   docker compose run --rm smoke
#   docker compose run --rm smoke-offline          # same run, network disabled
#
# Overrides (environment):
#   SCENE   default /data/shudian
#   OUT     default /outputs/smoke
#   ITERS   default 200
set -euo pipefail

SCENE="${SCENE:-/data/shudian}"
OUT="${OUT:-/outputs/smoke}"
ITERS="${ITERS:-200}"

if [[ ! -d "${SCENE}/sparse" && ! -f "${SCENE}/transforms_train.json" ]]; then
  echo "FATAL: '${SCENE}' is neither a COLMAP scene (sparse/) nor a Blender scene" >&2
  echo "       (transforms_train.json). Mount the dataset at /data and set SCENE." >&2
  exit 1
fi

echo "[smoke] --- import probe ---------------------------------------------"
python - <<'PY'
import importlib

import torch

for module in ("diff_triangle_rasterization", "simple_knn._C", "rdel",
               "triangle_renderer", "scene", "utils.render_utils"):
    importlib.import_module(module)

print(f"[smoke] imports ok | torch {torch.__version__} | cuda {torch.version.cuda} | "
      f"{torch.cuda.get_device_name(0)}")
PY

echo "[smoke] --- training ${ITERS} iterations on ${SCENE} -----------------"
rm -rf "${OUT}"
python train.py \
  -s "${SCENE}" \
  -m "${OUT}" \
  --eval \
  --quiet \
  --iterations "${ITERS}" \
  --test_iterations "${ITERS}" \
  --save_iterations "${ITERS}"

CKPT="${OUT}/point_cloud/iteration_${ITERS}/point_cloud_state_dict.pt"
if [[ ! -f "${CKPT}" ]]; then
  echo "FATAL: expected checkpoint '${CKPT}' was not written" >&2
  exit 1
fi

echo "[smoke] PASS: ${CKPT} ($(du -h "${CKPT}" | cut -f1))"
