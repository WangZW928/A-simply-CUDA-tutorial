#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
RESULTS_DIR="${PROJECT_ROOT}/results"
LOG_DIR="${RESULTS_DIR}/logs"
BUILD_ROOT="${PROJECT_ROOT}/build_sweep"
COMBINED_CSV="${RESULTS_DIR}/reg_occ_sweep.csv"

TMP_SIZES=${TMP_SIZES:-"8 16 24 32 48 64 80 96 128"}
REPEATS=${REPEATS:-20}
WARMUP=${WARMUP:-5}
ITERS=${ITERS:-256}

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${BUILD_ROOT}"

echo "[sweep] tmp sizes: ${TMP_SIZES}"
echo "[sweep] repeats=${REPEATS} warmup=${WARMUP} iters=${ITERS}"

echo "device_name,sm_count,kernel,high_reg_tmp_size,threads_per_block,blocks,elements,repeats,warmup,iters,regs_per_thread,shmem_static_bytes,max_active_blocks_per_sm,active_warps_per_sm,max_warps_per_sm,theoretical_occupancy,avg_ms,std_ms,elems_per_ms" > "${COMBINED_CSV}"

for sz in ${TMP_SIZES}; do
  BUILD_DIR="${BUILD_ROOT}/tmp_${sz}"
  OUT_CSV="${RESULTS_DIR}/reg_occ_tmp_${sz}.csv"
  LOG_FILE="${LOG_DIR}/build_run_tmp_${sz}.log"

  echo "[sweep] ===== HIGH_REG_TMP_SIZE=${sz} =====" | tee "${LOG_FILE}"
  rm -rf "${BUILD_DIR}"

  {
    cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_CUDA_FLAGS="-DHIGH_REG_TMP_SIZE=${sz}"
    cmake --build "${BUILD_DIR}" -j
    "${BUILD_DIR}/reg_occ_bench" "${OUT_CSV}" "${REPEATS}" "${WARMUP}" "${ITERS}"
  } >> "${LOG_FILE}" 2>&1

  tail -n +2 "${OUT_CSV}" >> "${COMBINED_CSV}"
  echo "[sweep] done size=${sz}, csv=${OUT_CSV}, log=${LOG_FILE}"
done

python3 "${PROJECT_ROOT}/scripts/plot_sweep.py" --input "${COMBINED_CSV}" --outdir "${RESULTS_DIR}"
echo "[sweep] all done. combined csv: ${COMBINED_CSV}"

