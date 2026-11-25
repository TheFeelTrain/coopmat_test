#!/usr/bin/env bash
# Runs the comparison three times (GPU clocks wander between runs on RDNA3;
# always look at the "min ms" column, and only trust numbers from an idle GPU).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GRID="${GRID:-8192}"
for i in 1 2 3; do
  echo "================ run $i ================"
  MANGOHUD=0 ./build/gemm_bench --grid "${GRID}" \
    build/gemm_coop.spv build/gemm_f32.spv build/gemm_f16.spv
done
