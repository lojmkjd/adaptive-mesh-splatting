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

shopt -s nullglob
renders=("${RENDER_DIR}/renders"/*.png)
gts=("${RENDER_DIR}/gt"/*.png)
if (( ${#renders[@]} == 0 )); then
  echo "FATAL: render.py wrote no images under '${RENDER_DIR}/renders'." >&2
  echo "       A model trained without --eval has no test split; retrain with --eval." >&2
  exit 1
fi
if (( ${#renders[@]} != ${#gts[@]} )); then
  echo "FATAL: ${#renders[@]} renders but ${#gts[@]} ground-truth images." >&2
  exit 1
fi
echo "[smoke-render] rendered ${#renders[@]} test views"

echo "[smoke-render] --- metrics.py ----------------------------------------"
python metrics.py -m "${MODEL}"

python - "${MODEL}" "${ITER}" <<'PY'
import json
import pathlib
import sys

model, iteration = pathlib.Path(sys.argv[1]), sys.argv[2]
results = model / "results.json"
if not results.is_file():
    sys.exit(f"FATAL: metrics.py did not write '{results}' (look for 'Unable to compute metrics')")

data = json.loads(results.read_text())
methods = data.get(str(model)) or data.get(sys.argv[1]) or next(iter(data.values()), {})
method = methods.get(f"ours_{iteration}") or next(iter(methods.values()), {})

missing = {"PSNR", "SSIM", "LPIPS"} - set(method)
if missing:
    sys.exit(f"FATAL: results.json is missing {sorted(missing)}: {method!r}")

psnr, ssim, lpips = (float(method[k]) for k in ("PSNR", "SSIM", "LPIPS"))
if not 1.0 < psnr < 100.0:
    sys.exit(f"FATAL: implausible PSNR {psnr:.3f} (test split mismatch?)")
if not 0.0 <= ssim <= 1.0 or not -1.0 <= lpips <= 2.0:
    sys.exit(f"FATAL: implausible SSIM/LPIPS {ssim:.4f}/{lpips:.4f}")

print(f"[smoke-render] PASS PSNR={psnr:.2f} SSIM={ssim:.4f} LPIPS={lpips:.4f}")
PY
