#!/usr/bin/env bash
#
# Render + metrics smoke: proves render.py and metrics.py on a real checkpoint.
# Training must have run first; this script never trains.
#
#   docker compose run --rm smoke            # produces /outputs/smoke (200 iters)
#   docker compose run --rm smoke-render     # this script
#   docker compose run --rm smoke-render-offline
#
# Overrides (environment):
#   MODEL   model directory      default /outputs/smoke
#   ITER    checkpoint iteration default 200
#   FULL    set to 1 to render the train split as well (slower)
set -euo pipefail

MODEL="${MODEL:-/outputs/smoke}"
ITER="${ITER:-200}"
FULL="${FULL:-0}"

CKPT="${MODEL}/point_cloud/iteration_${ITER}/point_cloud_state_dict.pt"
if [[ ! -f "${CKPT}" ]]; then
  echo "FATAL: no checkpoint at '${CKPT}'." >&2
  echo "       Run 'docker compose run --rm smoke' first, or set MODEL/ITER." >&2
  exit 1
fi

RENDER_DIR="${MODEL}/test/ours_${ITER}"
rm -rf "${RENDER_DIR}" "${MODEL}/results.json" "${MODEL}/per_view.json"

echo "[smoke-render] --- render.py at iteration ${ITER} ---------------------"
if [[ "${FULL}" == "1" ]]; then
  python render.py -m "${MODEL}" --iteration "${ITER}" --quiet
else
  python render.py -m "${MODEL}" --iteration "${ITER}" --skip_train --quiet
fi

python docker/verify_outputs.py renders "${MODEL}" "${ITER}"

echo "[smoke-render] --- metrics.py ----------------------------------------"
python metrics.py -m "${MODEL}"

python docker/verify_outputs.py metrics "${MODEL}" "${ITER}"
