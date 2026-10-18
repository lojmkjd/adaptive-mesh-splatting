#!/usr/bin/env python3
"""Render/metrics assertions shared by docker/smoke-render.sh and docker/smoke-guard.sh.

They live in Python rather than inline shell so the regression suite can drive
each failure path with a fixture directory instead of a real training run.
metrics.py swallows its own exceptions and only prints "Unable to compute
metrics", so its exit code proves nothing; these checks are the actual proof.

    python docker/verify_outputs.py renders <model_dir> <iteration>
    python docker/verify_outputs.py metrics <model_dir> <iteration>
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

METRICS = ("PSNR", "SSIM", "LPIPS")


def fail(message: str) -> "NoReturn":
    print(f"FATAL: {message}", file=sys.stderr)
    raise SystemExit(1)


def check_renders(model: pathlib.Path, iteration: str) -> None:
    """One render and one ground truth per test view, and at least one view."""
    render_dir = model / "test" / f"ours_{iteration}"
    renders = sorted((render_dir / "renders").glob("*.png"))
    gts = sorted((render_dir / "gt").glob("*.png"))
    if not renders:
        fail(
            f"render.py wrote no images under '{render_dir}/renders'.\n"
            "       A model trained without --eval has no test split; retrain with --eval."
        )
    if len(renders) != len(gts):
        fail(f"{len(renders)} renders but {len(gts)} ground-truth images.")
    print(f"[smoke-render] rendered {len(renders)} test views")


def find_stats(node: object, iteration: str) -> dict | None:
    """metrics.py writes {method: {PSNR, SSIM, LPIPS}} - it dumps full_dict[scene],
    not full_dict, so there is no scene key. Tolerate a scene-keyed wrapper too and
    prefer the requested method over whatever comes first."""
    if not isinstance(node, dict):
        return None
    if set(METRICS) <= set(node):
        return node
    preferred = node.get(f"ours_{iteration}")
    if isinstance(preferred, dict) and set(METRICS) <= set(preferred):
        return preferred
    for value in node.values():
        found = find_stats(value, iteration)
        if found is not None:
            return found
    return None


def check_metrics(model: pathlib.Path, iteration: str) -> None:
    """Require results.json with a complete metric block holding plausible values."""
    results = model / "results.json"
    if not results.is_file():
        fail(f"metrics.py did not write '{results}' (look for 'Unable to compute metrics')")

    data = json.loads(results.read_text())
    stats = find_stats(data, iteration)
    if stats is None:
        fail(f"no PSNR/SSIM/LPIPS block in '{results}': {json.dumps(data)[:400]}")

    psnr, ssim, lpips = (float(stats[key]) for key in METRICS)
    if not 1.0 < psnr < 100.0:
        fail(f"implausible PSNR {psnr:.3f} (test split mismatch?)")
    if not 0.0 <= ssim <= 1.0 or not -1.0 <= lpips <= 2.0:
        fail(f"implausible SSIM/LPIPS {ssim:.4f}/{lpips:.4f}")

    print(f"[smoke-render] PASS PSNR={psnr:.2f} SSIM={ssim:.4f} LPIPS={lpips:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assert render/metrics smoke output.")
    parser.add_argument("check", choices=("renders", "metrics"))
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("iteration")
    args = parser.parse_args()

    if args.check == "renders":
        check_renders(args.model, args.iteration)
    else:
        check_metrics(args.model, args.iteration)


if __name__ == "__main__":
    main()
