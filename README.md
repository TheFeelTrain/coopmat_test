# Cooperative matrix micro test

This is a small test to show that `VK_KHR_cooperative_matrix` is faster
than a plain shader for fp16 matrix multiply with fp32 accumulate.

It builds the same shader three ways. Only the 16x16x16 multiply changes.
The rest of the code is the same in all three.

| build | how it does the multiply |
|---|---|
| `gemm_coop` | `coopMatMulAdd`, fp16 in, fp32 accumulate |
| `gemm_f32` | plain dot products, fp32 accumulate |
| `gemm_f16` | plain dot products, fp16 accumulate |

`gemm_coop` and `gemm_f32` give the same answers. `gemm_f16` is less
accurate but is included to show that even the fast plain form loses.

## Results

Test card: Radeon RX 7900 XTX, RADV, Mesa 26.3.0-devel.

Each dispatch is 17.2 GFLOP. 8192 workgroups. Times are per dispatch
on the GPU. 300 warmup dispatches run before timing starts.

| test | best time | speed | share of fp16 peak | max error | rms error |
|---|---|---|---|---|---|
| `gemm_coop` | 0.335 ms | 51.3 TFLOPS | 45.3 percent | 7.8e-03 | 1.7e-03 |
| `gemm_f32` | 0.924 ms | 18.6 TFLOPS | 16.4 percent | 7.8e-03 | 1.7e-03 |
| `gemm_f16` | 0.800 ms | 21.5 TFLOPS | 19.0 percent | 2.3e-02 | 5.2e-03 |

Peak used here is 113.2 TFLOPS, which is the rated fp16 speed of a
7900 XTX at 2.30 GHz. You can pass `--peak` to set your own card value.

Coop is 2.7x faster than plain with the same accuracy, and 2.4x faster
than plain with lower accuracy. Errors are checked against a
double precision CPU result. The small error that is left is just
fp16 output rounding.

## Why plain is slower here

On RDNA3 the fast matrix path does 4096 MACs per step at full fp16
speed with fp32 accumulate. A plain shader that keeps fp32 accumulate
needs one instruction per MAC, so its limit is half that. There is a
packed dot instruction that does two per step, but plain GLSL cannot
use it with fp32 accumulate.

The test also folds four 16x16 results together after the multiply.
That fold needs all four results for one output point in the same
lane. On the plain path that leaves 4 outputs per lane, in a 2x2
block. With blocks that small, half the steps are loads instead of
math. The coop path does not have this limit because the matrix
layout is spread across the lanes.

Links:

- https://hipfire.dev/learn/rdna-wmma/
- https://gpuopen.com/learn/wmma_on_rdna3/
- https://www.phoronix.com/news/RADV-VK_KHR_cooperative_matrix

## Build and run

```sh
./build.sh
./run.sh
```

`build.sh` needs `glslc`, `g++`, and the Vulkan headers and library.
Nothing else.

To run by hand:

```sh
./build/gemm_bench --grid 8192 build/gemm_coop.spv build/gemm_f32.spv build/gemm_f16.spv
```

Options:

```
--grid N     workgroup count (default 8192)
--iters N    timed runs per test (default 40)
--slots N    distinct B inputs to cycle through (default 64)
--peak F     peak TFLOPS used for the percent column (default 113.2)
```

If the card or driver has no `VK_KHR_cooperative_matrix`, or no
16x16x16 fp16 to fp32 form, the coop test is skipped and the plain
tests still run. The card name and the list of supported matrix
forms print first.

Run it three times on a quiet GPU and take the best time. Clocks
move around between runs, so the best time is the one to trust.

## Files

| file | what it is |
|---|---|
| `shaders/gemm.comp` | the kernel. `USE_COOPMAT` and `PLAIN_F16` pick the multiply. The rest is shared. |
| `gemm_bench.cpp` | host code. Sets up Vulkan, times each dispatch, checks answers. |
| `vk_bench.h` | small Vulkan setup helpers. No extra deps besides Vulkan. |
| `results.txt` | output from the last `run.sh`. |

## What the test does

Each workgroup does 1,048,576 MACs:

- 4 blocks along k, each 16 deep
- times 4 tile groups times 2 k groups
- times 4 separate 16x16x16 multiplies

Input A comes from global memory. Input B is copied to local memory
first with a row stride of 24 fp16 values. The stride of 24 avoids
local memory bank conflicts. Both plain and coop paths read the same
bytes.

After the multiplies, the four results are added and subtracted
together into 8 fp32 sums. Those sums are stored as fp16 and written
out as fp16x4. All three builds do this part the same way.
