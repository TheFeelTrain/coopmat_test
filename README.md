# Does VK_KHR_cooperative_matrix actually win? (RDNA3, fp16 GEMM with fp32 accumulate)

A ~300-line reproducer for the thing we disagree about.  Same kernel, three
builds of the same source, only the 16x16x16 matmul differs:

| build | matmul |
|---|---|
| `gemm_coop` | 4x `coopMatMulAdd` (VK_KHR_cooperative_matrix, fp16 in / **fp32** accumulate) |
| `gemm_f32`  | dot products, **fp32** accumulate (the numerically equivalent plain shader) |
| `gemm_f16`  | dot products, **fp16** accumulate (the fastest plain shader, but 3x the error) |

Everything else is identical source: same workgroup shape (2 waves x 64 lanes),
same LDS staging, same padded LDS stride, same fp32 Winograd-style "fold", same
epilogue, same dispatch.

## Results

Radeon RX 7900 XTX (RADV NAVI31, gfx1100), Mesa 26.3.0-devel, glslc 2026.4.
17.2 GFLOP per dispatch, grid = 8192 workgroups, per-dispatch GPU timestamps,
300 priming dispatches before measuring.  Three runs, `min ms` per run:

| variant | min ms, best run | TFLOP/s | % of fp16 peak¹ | max err vs fp64 CPU ref | rms err |
|---|---|---|---|---|---|
| `gemm_coop` | **0.339** (other runs .339 / .340) | 50.7 | 44.8% | 7.8e-3 | 1.7e-3 |
| `gemm_f32`  | **0.920** (other runs .921 / .922) | 18.7 | 16.5% | 7.8e-3 | 1.7e-3 |
| `gemm_f16`  | **0.800** (other runs .800 / .801) | 21.5 | 19.0% | 2.3e-2 | 5.2e-3 |

¹ 113.2 TFLOP/s = fp16 nameplate of the 7900 XTX at its 2.30 GHz max clock
(96 CU x 128 lanes x 2 MAC/lane/clk x 2).  `--peak` lets you override it.

**Speedups: coop vs plain-fp32-acc = 2.72x, coop vs plain-fp16-acc = 2.36x.**
The fp32-accumulate ones are the apples-to-apples pair (identical error, both
match a double-precision CPU reference to the same 7.8e-3, which is just the
fp16 output rounding).  `gemm_f16` gives up 3x the accuracy and still loses.

## Why the gap is that large (and why it is not a strawman)

1. **fp32 accumulation has no packed form on RDNA3.**  `v_wmma_f32_16x16x16_f16`
   does 4096 MACs per issue at the full fp16 rate (that is why the WMMA roof is
   the same ~123 TFLOP/s as packed fp16).  A plain shader accumulating into f32
   needs one `v_fma_mix_f32` per MAC, i.e. one instruction per MAC, so its
   hard ceiling is *half* the fp16 peak.  The only 2-MAC-per-instruction
   fp16 path that accumulates in f32 is `v_dot2_f32_f16`, and GLSL cannot
   express it (a `f16vec2` `dot()` has an fp16 result, so it compiles to
   `v_dot2_f16_f16`).

2. **The Winograd fold pins the plain shader to 2x2 per-lane tiles.**  The fold
   is `s0 = C0+C1+C2`, `s1 = C1-C2-C3` followed by adds/subtracts into four f32
   accumulators.  All four 16x16 GEMM results for a given output element must be
   in the same lane.  With 64 lanes and a 16x16 output that leaves exactly 4
   output elements per lane, i.e. a 2x2 tile, and a 2x2 tile with 4 independent
   matmuls needs 4 operand loads per 8 MACs - half of the plain shader's issue
   slots go to loads, so it lands at ~1 MAC/instruction even in fp16-accumulate
   mode.  (The `coopmat` path has no such constraint: the 16x16 identity lives
   in the lane layout, so `FOLD4` is just element-wise math on accumulators.)

   A 4x4-tile plain variant with the fold moved into LDS was analyzed: it halves
   the loads (64 vs 128 per group) but adds 4 LDS stores + 2 barriers + ~32 LDS
   ops per group, i.e. roughly a wash, which is why it is not here.

3. The absolute numbers agree with that model.  `coop` runs at 44% of the fp16
   peak, i.e. ~55% of its own issue budget is the matmul and the rest is the
   fold + fragment loads - which is why the production kernel (V transform,
   strip loads and writeback added) sits around 42% of peak.  `gemm_f32` reaches
   33% of even its own 1-MAC/instruction ceiling: 128 operand loads per 256 FMAs
   plus L2 latency at 5 workgroups/CU (LDS-limited) is what eats the rest.  The
   plain path is not a strawman with a silly tiling - it is stuck behind fp32
   accumulation and the fold, and both of those are this kernel's design.

## Why testing this on RDNA2 cannot show a win

RDNA2 (`gfx1030`) has **no matrix unit** - it tops out at `v_dot4`/`v_dot2`
packed dot products.  WMMA (`v_wmma_f32_16x16x16_f16`) arrived with RDNA3
(`gfx1100`), and RADV's `VK_KHR_cooperative_matrix` support is built on those
RDNA3+ WMMA instructions.  On RDNA2 there is nothing for the extension to map
to, so any synthetic test there measures a fallback, not cooperative matrices.
If the RDNA2 test happened to run at all, it was not measuring the same thing
this benchmark measures.

References:
- <https://hipfire.dev/learn/rdna-wmma/> - "RDNA2 gfx1030: v_dot4 / v_dot2 (packed DOT, **no WMMA**)", RDNA3 16x16x16 WMMA, WMMA roof ~123 TF vs dot2 roof.
- <https://gpuopen.com/learn/wmma_on_rdna3/> - WMMA is an RDNA3 feature.
- <https://www.phoronix.com/news/RADV-VK_KHR_cooperative_matrix> - RADV coop matrix "relies on WMMA instructions ... found with RDNA3 GPUs and newer".

The extension is genuinely awkward, and we work around two RADV bugs in the
real kernel (wave32 `coopMatStore` misplacement, `coopMatLoad` of B from global
for the upper fragment half) - but awkward is not the same as slow on hardware
that has the matrix unit.

## Build / run

```sh
./build.sh     # glslc x3 + g++ (needs libvulkan + vulkan headers only)
./run.sh       # 3 runs of all three variants
# or explicitly:
./build/gemm_bench --grid 8192 build/gemm_coop.spv build/gemm_f32.spv build/gemm_f16.spv
```

Exit behaviour on other hardware: the device list and the available
coop-matrix configurations are printed first; if `VK_KHR_cooperative_matrix` (or
the 16x16x16 fp16->fp32 config) is missing, the coop variant is skipped with a
message and the plain ones still run.  Use `--peak <TFLOPS>` to fix the % of
peak column for another GPU.

```
--grid N     workgroup count (default 8192 = 17.2 GFLOP)
--iters N    timed dispatches per variant (default 40)
--slots N    distinct B slots in the input buffer (default 64)
--peak F     peak TFLOP/s used for the % column (default 113.2)
```

## Files

| file | what |
|---|---|
| `shaders/gemm.comp` | the whole kernel.  `#ifdef USE_COOPMAT` / `#ifdef PLAIN_F16` select the matmul; the fold, staging and epilogue are shared code. |
| `gemm_bench.cpp` | headless Vulkan harness: buffers, dispatch, GPU timestamps, fp64 CPU reference, fp16 helpers. |
| `vk_bench.h` | minimal Vulkan 1.4 compute helpers. |
| `results.txt` | raw output of the last `run.sh`. |

## Workload details (so you can sanity-check the shape)

Per workgroup, exactly the inner GEMM of our Winograd F(2x2,3x3) conv kernel:

- 4 "c-blocks" (the k chain, 16 input channels each) x 4 wpi x 2 kb groups
  x 4 independent 16x16x16 matmuls = **1,048,576 MACs**
- A ("U" weights) read from global, 128 KB total, L2-resident, same layout as
  the real kernel.  B ("V") staged into 12 KB of LDS with the real kernel's
  padded row stride of 24 fp16, read by `coopMatLoad` ColumnMajor / by
  `f16vec2` pairs for the plain path (same bytes, same LDS footprint).
  ColumnMajor was measured faster than RowMajor for this B fragment load in
  this layout, so the coop variant is not being handicapped by it.
- The fold, the 8 fp32 accumulator matrices (4 per kb set) and the fp16
  `ColumnMajor` y-staging + vec4 writeback of the real kernel's epilogue are
  reproduced, minus the V transform and the bias/silu (identical work for all
  three variants, and they only dilute the ratio).
