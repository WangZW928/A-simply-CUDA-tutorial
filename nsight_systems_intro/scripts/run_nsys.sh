#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CUDA_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
STREAM_DIR="${CUDA_ROOT}/cuda_stream_intro"

OUT_DIR="${PROJECT_DIR}/results"
OUT_NAME="${OUT_NAME:-stream_overlap}"
REPEATS="${REPEATS:-10}"
WARMUP="${WARMUP:-2}"
CHUNK_ELEMS="${CHUNK_ELEMS:-1048576}"
ITERS="${ITERS:-512}"

mkdir -p "${OUT_DIR}"

if ! command -v nsys >/dev/null 2>&1; then
    echo "error: nsys not found in PATH" >&2
    exit 1
fi

if [[ ! -x "${STREAM_DIR}/build/stream_bench" ]]; then
    cmake -S "${STREAM_DIR}" -B "${STREAM_DIR}/build"
    cmake --build "${STREAM_DIR}/build" -j
fi

CSV_PATH="${OUT_DIR}/stream_profile_input.csv"
REPORT_PATH="${OUT_DIR}/${OUT_NAME}"

nsys profile \
    --trace=cuda,osrt,nvtx \
    --sample=none \
    --force-overwrite=true \
    --output="${REPORT_PATH}" \
    "${STREAM_DIR}/build/stream_bench" "${CSV_PATH}" "${REPEATS}" "${WARMUP}" "${CHUNK_ELEMS}" "${ITERS}"

echo "Nsight Systems report: ${REPORT_PATH}.nsys-rep"
echo "Benchmark CSV: ${CSV_PATH}"
echo "Open with: nsys-ui ${REPORT_PATH}.nsys-rep"
