#!/usr/bin/env bash
#
# Guard regression suite: proves the smoke-render assertions fail loudly instead
# of letting a broken render or a silent metrics.py failure look like a pass.
# Fixtures only - no dataset, no GPU work, no network, about a second.
#
#   docker compose run --rm smoke-guard
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

VERIFY=(python docker/verify_outputs.py)
passed=0
regressed=0

expect_failure() {
  local desc="${1}" needle="${2}"
  shift 2
  local out rc
  out="$("$@" 2>&1)"
  rc=$?
  if (( rc == 0 )); then
    echo "FAIL [${desc}] expected a non-zero exit, got 0" >&2
    regressed=$((regressed + 1))
    return
  fi
  if [[ "${out}" != *"${needle}"* ]]; then
    echo "FAIL [${desc}] expected '${needle}' in the output, got:" >&2
    echo "${out}" >&2
    regressed=$((regressed + 1))
    return
  fi
  echo "ok   [${desc}] exit=${rc} ${needle}"
  passed=$((passed + 1))
}

expect_success() {
  local desc="${1}" needle="${2}"
  shift 2
  local out rc
  out="$("$@" 2>&1)"
  rc=$?
  if (( rc != 0 )); then
    echo "FAIL [${desc}] expected exit 0, got ${rc}:" >&2
    echo "${out}" >&2
    regressed=$((regressed + 1))
    return
  fi
  if [[ "${out}" != *"${needle}"* ]]; then
    echo "FAIL [${desc}] expected '${needle}' in the output, got:" >&2
    echo "${out}" >&2
    regressed=$((regressed + 1))
    return
  fi
  echo "ok   [${desc}] exit=0 ${needle}"
  passed=$((passed + 1))
}

# A stand-in for a trained model: only the checkpoint file is ever touched here.
checkpoint() {
  mkdir -p "${1}/point_cloud/iteration_200"
  : >"${1}/point_cloud/iteration_200/point_cloud_state_dict.pt"
}

write_results() {
  printf '%s\n' "${2}" >"${1}/results.json"
}

echo "[smoke-guard] --- failure paths --------------------------------------"

# G1 missing checkpoint: smoke-render.sh must refuse before it calls render.py,
# which is also why this case needs no GPU.
mkdir -p "${WORK}/no-checkpoint"
expect_failure "G1 missing checkpoint" "no checkpoint" \
  env MODEL="${WORK}/no-checkpoint" ITER=200 bash docker/smoke-render.sh

# G2 a model trained without --eval has no test split, so render.py writes nothing.
no_test="${WORK}/no-test"
checkpoint "${no_test}"
mkdir -p "${no_test}/test/ours_200/renders" "${no_test}/test/ours_200/gt"
expect_failure "G2 no test split" "wrote no images" \
  "${VERIFY[@]}" renders "${no_test}" 200

# G3 renders and ground truth must agree, or the metrics compare different views.
mismatch="${WORK}/mismatch"
checkpoint "${mismatch}"
mkdir -p "${mismatch}/test/ours_200/renders" "${mismatch}/test/ours_200/gt"
: >"${mismatch}/test/ours_200/renders/r000.png"
: >"${mismatch}/test/ours_200/renders/r001.png"
: >"${mismatch}/test/ours_200/gt/g000.png"
: >"${mismatch}/test/ours_200/gt/g001.png"
: >"${mismatch}/test/ours_200/gt/g002.png"
expect_failure "G3 render/gt count mismatch" "ground-truth images" \
  "${VERIFY[@]}" renders "${mismatch}" 200

# G4 metrics.py swallows its exceptions, so a missing file is the only signal.
no_results="${WORK}/no-results"
checkpoint "${no_results}"
expect_failure "G4 metrics.py wrote nothing" "did not write" \
  "${VERIFY[@]}" metrics "${no_results}" 200

# G5 an incomplete metric block must not be read as a result.
partial="${WORK}/partial"
checkpoint "${partial}"
write_results "${partial}" '{"ours_200": {"PSNR": 20.0, "SSIM": 0.6}}'
expect_failure "G5 incomplete metric block" "no PSNR/SSIM/LPIPS block" \
  "${VERIFY[@]}" metrics "${partial}" 200

# G6 values outside the plausible range mean the wrong split was scored.
implausible="${WORK}/implausible"
checkpoint "${implausible}"
write_results "${implausible}" '{"ours_200": {"PSNR": 0.0, "SSIM": 0.6, "LPIPS": 0.3}}'
expect_failure "G6 implausible PSNR" "implausible PSNR" \
  "${VERIFY[@]}" metrics "${implausible}" 200

echo "[smoke-guard] --- accepted shapes -----------------------------------"

# G7 the real shape metrics.py writes: {method: {PSNR, SSIM, LPIPS}}.
good="${WORK}/good"
checkpoint "${good}"
write_results "${good}" '{"ours_200": {"SSIM": 0.2144, "PSNR": 13.41, "LPIPS": 0.7162}}'
expect_success "G7 results.json as metrics.py writes it" "PASS PSNR=13.41" \
  "${VERIFY[@]}" metrics "${good}" 200

# G8 a scene-keyed wrapper stays accepted, in case metrics.py output ever nests.
wrapped="${WORK}/wrapped"
checkpoint "${wrapped}"
write_results "${wrapped}" '{"/outputs/smoke": {"ours_200": {"SSIM": 0.2144, "PSNR": 13.41, "LPIPS": 0.7162}}}'
expect_success "G8 scene-keyed results.json" "PASS PSNR=13.41" \
  "${VERIFY[@]}" metrics "${wrapped}" 200

# G9 matching render and ground-truth counts pass.
rendered="${WORK}/rendered"
checkpoint "${rendered}"
mkdir -p "${rendered}/test/ours_200/renders" "${rendered}/test/ours_200/gt"
: >"${rendered}/test/ours_200/renders/r000.png"
: >"${rendered}/test/ours_200/gt/g000.png"
expect_success "G9 matching render/gt counts" "rendered 1 test views" \
  "${VERIFY[@]}" renders "${rendered}" 200

echo
if (( regressed > 0 )); then
  echo "[smoke-guard] FAIL: ${regressed} guard(s) regressed, ${passed} passed" >&2
  exit 1
fi
echo "[smoke-guard] PASS: ${passed} guards behave as documented"
