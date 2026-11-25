#!/usr/bin/env bash
# Run the test three times. GPU clocks drift, so use the best time.
# Run on a quiet GPU with nothing else using it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GRID="${GRID:-8192}"
for i in 1 2 3; do
  echo "================ run $i ================"
  MANGOHUD=0 ./build/gemm_bench --grid "${GRID}" \
    build/gemm_coop.spv build/gemm_f32.spv build/gemm_f16.spv
done
