#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CUDA_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
TARGET_DIR="${TARGET_DIR:-${CUDA_ROOT}/memory_coalescing_intro}"
TARGET_BIN="${TARGET_BIN:-${TARGET_DIR}/build/mem_coalescing_bench}"

OUT_DIR="${PROJECT_DIR}/results"
OUT_NAME="${OUT_NAME:-mem_coalescing_ncu}"
REPEATS="${REPEATS:-5}"
WARMUP="${WARMUP:-1}"
ELEMENTS="${ELEMENTS:-1048576}"
MAX_STRIDE="${MAX_STRIDE:-32}"

mkdir -p "${OUT_DIR}"

if ! command -v ncu >/dev/null 2>&1; then
    echo "error: ncu not found in PATH" >&2
    exit 1
fi

if [[ ! -x "${TARGET_BIN}" ]]; then
    cmake -S "${TARGET_DIR}" -B "${TARGET_DIR}/build"
    cmake --build "${TARGET_DIR}/build" -j
fi

ncu \
    --set full \
    --target-processes all \
    --kernel-name-base demangled \
    --export "${OUT_DIR}/${OUT_NAME}" \
    --force-overwrite \
    "${TARGET_BIN}" "${OUT_DIR}/mem_coalescing_profile_input.csv" "${REPEATS}" "${WARMUP}" "${ELEMENTS}" "${MAX_STRIDE}"

echo "Nsight Compute report: ${OUT_DIR}/${OUT_NAME}.ncu-rep"
echo "Open with: ncu-ui ${OUT_DIR}/${OUT_NAME}.ncu-rep"
