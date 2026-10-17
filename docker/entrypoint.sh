#!/usr/bin/env bash
#
# Container entrypoint: proves the GPU is reachable before running the command,
# so a missing device reservation fails with one clear line instead of a torch
# stack trace deep inside training.
#
# SKIP_GPU_CHECK=1 bypasses the probe for CPU-only commands.
set -euo pipefail

if [[ "${SKIP_GPU_CHECK:-0}" != "1" ]]; then
  python - <<'PY'
import sys

import torch

if not torch.cuda.is_available():
    sys.exit(
        "FATAL: no CUDA device is visible inside the container.\n"
        "  * docker run: add --gpus all\n"
        "  * docker compose: keep the deploy.resources.reservations.devices entry\n"
        "  * host: the driver must support the cu126 runtime (driver >= 560)\n"
        "Set SKIP_GPU_CHECK=1 to run CPU-only commands anyway."
    )

name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info()
print(
    f"[entrypoint] torch {torch.__version__} | cuda {torch.version.cuda} | "
    f"{name} | {total / 2**30:.1f} GiB total, {free / 2**30:.1f} GiB free",
    flush=True,
)
PY
fi

exec "$@"
