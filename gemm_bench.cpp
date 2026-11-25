// Coop vs plain shader test.
// Usage: gemm_bench [--grid N] [--iters N] [--slots N] [--peak F] a.spv [b.spv ...]
//
// Each input runs the same work (grid x 1,048,576 MACs). For each one it
// prints time per dispatch, TFLOPS, and error against a CPU result.
#include "vk_bench.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <vector>

// Float to half and back. Used to make test data and check results.
static uint16_t f2h(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = x & 0x7fffffu;
  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;
    mant |= 0x800000u;
    const int shift = 14 - exp;
    uint32_t m = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (m & 1u))) ++m;
    return (uint16_t)(sign | m);
  }
  if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
  mant += 0x1000u;
  if (mant & 0x800000u) {
    mant = 0;
    if (++exp >= 31) return (uint16_t)(sign | 0x7c00u);
  }
  return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static float h2f(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      int e = -1;
      do {
        ++e;
        mant <<= 1;
      } while ((mant & 0x400u) == 0);
      mant &= 0x3ffu;
      out = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    out = sign | 0x7f800000u | (mant << 13);
  } else {
    out = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &out, 4);
  return f;
}

// These values match shaders/gemm.comp.
static constexpr uint32_t kNCB = 4;     // k blocks
static constexpr uint32_t kNGroup = 8;  // 4 tile groups x 2 k groups
static constexpr uint32_t kNMT = 4;     // multiplies per group
static constexpr uint32_t kWPV = 16;    // B matrices per k block
static constexpr uint32_t kVSTR = 24;   // B row stride in fp16
static constexpr uint32_t kSlotElems = kWPV * 16 * kVSTR;  // 6144 fp16 per B slot
static constexpr uint32_t kOutPerWG = 4096;  // fp16 values written per workgroup
// MACs per workgroup: 2 waves, each runs the full k chain.
static constexpr uint64_t kMacPerWG = 2ull * (uint64_t)kNCB * kNGroup * kNMT * 16 * 16 * 16;

// A has [wave][block][group][j][ko][c], 2*4*8*4*256 fp16 = 128 KB.
static constexpr uint32_t kAElems = 2 * kNCB * kNGroup * kNMT * 256;

struct VariantResult {
  std::string name;
  std::vector<double> ms;
  std::vector<uint16_t> out64;  // first 64 workgroups
  double max_err_cpu = 0.0;
  double rms_err_cpu = 0.0;
  double max_diff_ref = 0.0;
};

// CPU check for one (workgroup, wave). Follows the shader math exactly.
static void cpu_reference(const std::vector<float>& A, const std::vector<float>& B, uint32_t nslot,
                          uint32_t wg, uint32_t wave, double y[2][4][16][16]) {
  const uint32_t slot = wg % nslot;
  const uint32_t bsrc = slot * kSlotElems;
  std::memset(y, 0, sizeof(double) * 2 * 4 * 16 * 16);
  for (uint32_t cb = 0; cb < kNCB; ++cb) {
    for (uint32_t wpi = 0; wpi < 4; ++wpi) {
      for (uint32_t kb = 0; kb < 2; ++kb) {
        const uint32_t g = wpi * 2 + kb;
        const uint32_t abase = ((wave * (kNCB * kNGroup) + cb * kNGroup + g) * kNMT) * 256;
        double s0[16][16], s1[16][16];
        double c[4][16][16];
        for (uint32_t j = 0; j < 4; ++j) {
          const uint32_t wp = wpi * 4 + j;
          for (uint32_t ko = 0; ko < 16; ++ko) {
            for (uint32_t nt = 0; nt < 16; ++nt) {
              double acc = 0.0;
              for (uint32_t cc = 0; cc < 16; ++cc) {
                const double av = A[abase + j * 256 + ko * 16 + cc];
                const double bv = B[bsrc + wp * 16 * kVSTR + nt * kVSTR + cc];
                acc += av * bv;
              }
              c[j][ko][nt] = acc;
            }
          }
        }
        for (uint32_t ko = 0; ko < 16; ++ko)
          for (uint32_t nt = 0; nt < 16; ++nt) {
            s0[ko][nt] = c[0][ko][nt] + c[1][ko][nt] + c[2][ko][nt];
            s1[ko][nt] = c[1][ko][nt] - c[2][ko][nt] - c[3][ko][nt];
          }
        // Same fold as the shader.
        for (uint32_t ko = 0; ko < 16; ++ko)
          for (uint32_t nt = 0; nt < 16; ++nt) {
            if (wpi == 0) {
              y[kb][0][ko][nt] += s0[ko][nt];
              y[kb][1][ko][nt] += s1[ko][nt];
            } else if (wpi == 1) {
              y[kb][0][ko][nt] += s0[ko][nt];
              y[kb][1][ko][nt] += s1[ko][nt];
              y[kb][2][ko][nt] += s0[ko][nt];
              y[kb][3][ko][nt] += s1[ko][nt];
            } else if (wpi == 2) {
              y[kb][0][ko][nt] += s0[ko][nt];
              y[kb][1][ko][nt] += s1[ko][nt];
              y[kb][2][ko][nt] -= s0[ko][nt];
              y[kb][3][ko][nt] -= s1[ko][nt];
            } else {
              y[kb][2][ko][nt] -= s0[ko][nt];
              y[kb][3][ko][nt] -= s1[ko][nt];
            }
          }
      }
    }
  }
}

int main(int argc, char** argv) {
  int grid = 8192, iters = 40, warmup = 25;
  uint32_t nslot = 64;
  double peak_tflops = 113.2;  // 7900 XTX fp16 speed at 2.30 GHz
  std::vector<std::string> spvs;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--grid" && i + 1 < argc) grid = std::atoi(argv[++i]);
    else if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (a == "--peak" && i + 1 < argc) peak_tflops = std::atof(argv[++i]);
    else if (a == "--slots" && i + 1 < argc) nslot = (uint32_t)std::atoi(argv[++i]);
    else if (a.rfind("--", 0) == 0) {
      std::fprintf(stderr, "unknown arg %s\n", a.c_str());
      return 2;
    } else spvs.push_back(a);
  }
  if (spvs.empty()) {
    std::fprintf(stderr,
                 "usage: %s [--grid N] [--iters N] [--slots N] [--peak TFLOPS] x.spv [y.spv ...]\n",
                 argv[0]);
    return 2;
  }

  VkBench vk;
  vk.init();

  // Make test data. Values are in [-1, 1).
  const VkDeviceSize a_bytes = (VkDeviceSize)kAElems * 2;
  const VkDeviceSize b_bytes = (VkDeviceSize)nslot * kSlotElems * 2;
  const VkDeviceSize o_bytes = (VkDeviceSize)grid * kOutPerWG * 2;

  std::vector<uint16_t> Ah(kAElems), Bh((size_t)nslot * kSlotElems);
  std::vector<float> Af(kAElems), Bf((size_t)nslot * kSlotElems);
  uint64_t rng = 0x243F6A8885A308D3ull;
  auto next = [&]() {
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return rng;
  };
  for (size_t i = 0; i < Ah.size(); ++i) {
    const float f = ((float)(next() >> 40) / (float)(1 << 24) - 0.5f) * 2.0f;
    Ah[i] = f2h(f);
    Af[i] = h2f(Ah[i]);
  }
  for (size_t i = 0; i < Bh.size(); ++i) {
    const float f = ((float)(next() >> 40) / (float)(1 << 24) - 0.5f) * 2.0f;
    Bh[i] = f2h(f);
    Bf[i] = h2f(Bh[i]);
  }

  VkBuf bufA = vk.create_buffer(a_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
  VkBuf bufB = vk.create_buffer(b_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, false);
  VkBuf bufO = vk.create_buffer(o_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, false);
  vk.upload(bufA, Ah.data(), a_bytes);
  vk.upload(bufB, Bh.data(), b_bytes);

  VkDescriptorSetLayout set_layout = vk.make_set_layout(3);
  VkPipelineLayout pipe_layout = vk.make_pipeline_layout(set_layout, 16);

  VkDescriptorSet set = vk.alloc_set(set_layout);
  std::vector<VkBuf*> bufs = {&bufA, &bufB, &bufO};
  vk.bind_buffers(set, bufs);

  struct PC {
    uint32_t nslot, pad0, pad1, pad2;
  } pc{nslot, 0, 0, 0};

  // Run each input file.
  std::vector<VariantResult> results;
  const uint32_t dump_wgs = 64;
  std::vector<uint16_t> host_out64((size_t)dump_wgs * kOutPerWG);

  for (const std::string& spv : spvs) {
    VariantResult r;
    r.name = spv;
    VkShaderModule mod = vk.load_shader(spv.c_str());
    VkPipeline pipe = vk.make_pipeline(mod, pipe_layout, 64);
    if (pipe == VK_NULL_HANDLE) {
      std::fprintf(stderr, "[bench] skipping %s\n", spv.c_str());
      vkDestroyShaderModule(vk.device, mod, nullptr);
      continue;
    }
    // Warm up first so GPU clocks settle before timing.
    vk.time_dispatches(pipe, pipe_layout, set, &pc, sizeof(pc), (uint32_t)grid, 1, 300);
    r.ms = vk.time_dispatches(pipe, pipe_layout, set, &pc, sizeof(pc), (uint32_t)grid, iters,
                              warmup);
    // Copy back the first 64 workgroups for the CPU check.
    VkBuf bufSmall = vk.create_buffer((VkDeviceSize)host_out64.size() * 2,
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      false);
    vk.begin_cmd();
    VkBufferCopy c{};
    c.size = (VkDeviceSize)host_out64.size() * 2;
    vkCmdCopyBuffer(vk.cmd, bufO.buffer, bufSmall.buffer, 1, &c);
    vk.end_cmd_submit_wait();
    vk.download(bufSmall, host_out64.data(), (VkDeviceSize)host_out64.size() * 2);
    vk.destroy_buffer(bufSmall);
    r.out64 = host_out64;
    results.push_back(std::move(r));
    vkDestroyPipeline(vk.device, pipe, nullptr);
    vkDestroyShaderModule(vk.device, mod, nullptr);
  }

  // Check workgroups 0..3, both waves, against the CPU result.
  for (auto& r : results) {
    double max_err = 0.0, sum_sq = 0.0;
    size_t cnt = 0;
    for (uint32_t wg = 0; wg < 4; ++wg) {
      for (uint32_t wave = 0; wave < 2; ++wave) {
        double y[2][4][16][16];
        cpu_reference(Af, Bf, nslot, wg, wave, y);
        for (uint32_t kb = 0; kb < 2; ++kb)
          for (uint32_t px = 0; px < 4; ++px)
            for (uint32_t nt = 0; nt < 16; ++nt)
              for (uint32_t ko = 0; ko < 16; ++ko) {
                const uint32_t idx =
                    ((wg * 16u + nt) * 4u + px) * 64u + (wave * 32u + kb * 16u + ko);
                const double got = h2f(r.out64[idx]);
                const double want = y[kb][px][ko][nt];
                const double d = std::fabs(got - want);
                max_err = std::max(max_err, d);
                sum_sq += d * d;
                ++cnt;
              }
      }
    }
    r.max_err_cpu = max_err;
    r.rms_err_cpu = std::sqrt(sum_sq / (double)cnt);
  }

  // Largest change from the first input, over all 64 workgroups.
  for (size_t i = 0; i < results.size(); ++i) {
    double md = 0.0;
    for (size_t k = 0; k < results[i].out64.size(); ++k) {
      const double d = std::fabs(h2f(results[i].out64[k]) - h2f(results[0].out64[k]));
      md = std::max(md, d);
    }
    results[i].max_diff_ref = md;
  }

  // Print the table.
  const double macs = (double)grid * (double)kMacPerWG;
  const double flops = 2.0 * macs;
  std::printf("\n");
  std::printf("workload: grid=%d x %llu MACs = %.3f GMAC (%.3f GFLOP), %d iters\n", grid,
              (unsigned long long)kMacPerWG, macs / 1e9, flops / 1e9, iters);
  std::printf("%-34s %9s %9s %9s %8s %8s %10s %10s\n", "variant", "min ms", "med ms", "TFLOPS",
              "%peak", "maxerr", "rms", "maxdiff");
  for (auto& r : results) {
    std::vector<double> s = r.ms;
    std::sort(s.begin(), s.end());
    const double mn = s.front();
    const double med = s[s.size() / 2];
    const char* base = std::strrchr(r.name.c_str(), '/');
    base = base ? base + 1 : r.name.c_str();
    std::printf("%-34s %9.4f %9.4f %9.2f %7.1f%% %8.2e %8.2e %10.2e\n", base, mn, med,
                flops / (mn * 1e-3) / 1e12, 100.0 * (flops / (mn * 1e-3) / 1e12) / peak_tflops,
                r.max_err_cpu, r.rms_err_cpu, r.max_diff_ref);
  }
  std::printf("\n");
  std::printf("maxerr/rms = error vs double precision CPU check (workgroups 0..3, both waves)\n");
  std::printf("maxdiff    = largest change vs the first input, over 64 workgroups\n");
  std::fflush(stdout);
  return 0;
}
