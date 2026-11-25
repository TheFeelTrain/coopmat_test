#!/usr/bin/env bash
# Builds the three shader variants + the benchmark harness.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p build
GLSLC="${GLSLC:-glslc}"
CXX="${CXX:-g++}"

# The matmul is the ONLY difference between these three builds.
"${GLSLC}" -O --target-env=vulkan1.3 -DUSE_COOPMAT shaders/gemm.comp -o build/gemm_coop.spv
"${GLSLC}" -O --target-env=vulkan1.3               shaders/gemm.comp -o build/gemm_f32.spv
"${GLSLC}" -O --target-env=vulkan1.3 -DPLAIN_F16   shaders/gemm.comp -o build/gemm_f16.spv

"${CXX}" -O2 -std=c++17 -Wall -Wextra -o build/gemm_bench gemm_bench.cpp -lvulkan
echo "built: build/gemm_{coop,f32,f16}.spv + build/gemm_bench"
