// Minimal headless Vulkan-1.4 compute harness (no dependency beyond libvulkan).
// Used by the cooperative-matrix vs plain-shader micro-benchmark.
#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define VKB_CHECK(x)                                                                  \
  do {                                                                               \
    VkResult _r = (x);                                                               \
    if (_r != VK_SUCCESS) {                                                          \
      std::fprintf(stderr, "[vk] %s -> VkResult %d at %s:%d\n", #x, (int)_r,          \
                   __FILE__, __LINE__);                                              \
      std::exit(2);                                                                  \
    }                                                                                \
  } while (0)

struct VkBuf {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  void* mapped = nullptr;
};

struct VkBench {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties props{};
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_family = 0;
  VkCommandPool cmd_pool = VK_NULL_HANDLE;
  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkPhysicalDeviceMemoryProperties mem_props{};
  uint32_t timestamp_valid_bits = 0;
  float timestamp_period = 1.0f;
  bool has_coopmat = false;

  // ---------------------------------------------------------------- init ----
  void init(bool need_coopmat, const char* device_name_filter = nullptr) {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "coopmat-micro";
    app.applicationVersion = 1;
    app.pEngineName = "none";
    app.apiVersion = VK_API_VERSION_1_4;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VKB_CHECK(vkCreateInstance(&ici, nullptr, &instance));

    uint32_t n = 0;
    VKB_CHECK(vkEnumeratePhysicalDevices(instance, &n, nullptr));
    std::vector<VkPhysicalDevice> devs(n);
    VKB_CHECK(vkEnumeratePhysicalDevices(instance, &n, devs.data()));

    physical = VK_NULL_HANDLE;
    for (VkPhysicalDevice d : devs) {
      VkPhysicalDeviceProperties p{};
      vkGetPhysicalDeviceProperties(d, &p);
      bool coop = has_device_ext(d, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
      const bool name_ok = (device_name_filter == nullptr) ||
                           (std::strstr(p.deviceName, device_name_filter) != nullptr);
      // Prefer a discrete GPU that has cooperative matrix; never pick llvmpipe.
      const bool discrete = (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
      if (name_ok && discrete && (!need_coopmat || coop)) {
        physical = d;
        break;
      }
      if (physical == VK_NULL_HANDLE && name_ok && p.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU &&
          (!need_coopmat || coop)) {
        physical = d;
      }
    }
    if (physical == VK_NULL_HANDLE) {
      std::fprintf(stderr, "[vk] no suitable physical device (coopmat=%d)\n", (int)need_coopmat);
      std::exit(2);
    }
    vkGetPhysicalDeviceProperties(physical, &props);
    vkGetPhysicalDeviceMemoryProperties(physical, &mem_props);
    has_coopmat = has_device_ext(physical, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

    // queue family with compute
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &qn, qs.data());
    queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < qn; ++i) {
      if ((qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && (qs[i].queueFlags & VK_QUEUE_TRANSFER_BIT)) {
        queue_family = i;
        timestamp_valid_bits = qs[i].timestampValidBits;
        break;
      }
    }
    if (queue_family == UINT32_MAX) {
      for (uint32_t i = 0; i < qn; ++i) {
        if (qs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
          queue_family = i;
          timestamp_valid_bits = qs[i].timestampValidBits;
          break;
        }
      }
    }
    if (queue_family == UINT32_MAX) {
      std::fprintf(stderr, "[vk] no compute queue\n");
      std::exit(2);
    }

    // ---- features ----
    VkPhysicalDeviceVulkan11Features f11{};
    f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    f11.storageBuffer16BitAccess = VK_TRUE;
    f11.uniformAndStorageBuffer16BitAccess = VK_TRUE;

    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.shaderFloat16 = VK_TRUE;
    f12.shaderInt8 = VK_TRUE;
    f11.pNext = &f12;

    VkPhysicalDeviceVulkan13Features f13{};
    f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f13.subgroupSizeControl = VK_TRUE;
    f13.computeFullSubgroups = VK_TRUE;
    f12.pNext = &f13;

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR fcoop{};
    fcoop.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    fcoop.cooperativeMatrix = VK_TRUE;
    f13.pNext = &fcoop;

    std::vector<const char*> exts;
    if (need_coopmat || has_coopmat) exts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    exts.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);

    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &f11;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)exts.size();
    dci.ppEnabledExtensionNames = exts.data();
    VKB_CHECK(vkCreateDevice(physical, &dci, nullptr, &device));
    vkGetDeviceQueue(device, queue_family, 0, &queue);

    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queue_family;
    VKB_CHECK(vkCreateCommandPool(device, &pci, nullptr, &cmd_pool));

    VkCommandBufferAllocateInfo cbi{};
    cbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbi.commandPool = cmd_pool;
    cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbi.commandBufferCount = 1;
    VKB_CHECK(vkAllocateCommandBuffers(device, &cbi, &cmd));

    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VKB_CHECK(vkCreateFence(device, &fci, nullptr, &fence));

    VkPhysicalDeviceProperties p{};
    vkGetPhysicalDeviceProperties(physical, &p);
    timestamp_period = p.limits.timestampPeriod;

    // report cooperative matrix configs
    if (has_coopmat) {
      auto props_khr = query_coop_props();
      std::printf("[vk] %s | %s | Vulkan %u.%u.%u\n", p.deviceName,
                  driver_name(physical).c_str(), VK_API_VERSION_MAJOR(p.apiVersion),
                  VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion));
      std::printf("[vk] VK_KHR_cooperative_matrix configs (%zu):\n", props_khr.size());
      for (const auto& c : props_khr) {
        if (c.scope == VK_SCOPE_SUBGROUP_KHR) {
          std::printf("       %ux%ux%u A=%d B=%d C=%d R=%d scope=subgroup%s%s\n", c.MSize,
                      c.NSize, c.KSize, (int)c.AType, (int)c.BType, (int)c.CType,
                      (int)c.ResultType,
                      (c.MSize == 16 && c.NSize == 16 && c.KSize == 16 &&
                       c.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                       c.CType == VK_COMPONENT_TYPE_FLOAT32_KHR) ? "   <== 16x16x16 fp16->fp32"
                                                                 : "",
                      "");
        }
      }
    } else {
      std::printf("[vk] %s | %s | NO VK_KHR_cooperative_matrix\n", p.deviceName,
                  driver_name(physical).c_str());
    }
    std::printf("[vk] timestampPeriod=%.3f ns, validBits=%u\n", timestamp_period,
                timestamp_valid_bits);
    std::fflush(stdout);
  }

  static std::string driver_name(VkPhysicalDevice d) {
    VkPhysicalDeviceDriverProperties dp{};
    dp.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &dp;
    vkGetPhysicalDeviceProperties2(d, &p2);
    return std::string(dp.driverName) + " " + dp.driverInfo;
  }

  std::vector<VkCooperativeMatrixPropertiesKHR> query_coop_props() {
    std::vector<VkCooperativeMatrixPropertiesKHR> out;
    if (!has_coopmat) return out;
    auto fn = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)vkGetInstanceProcAddr(
        instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");
    if (!fn) return out;
    uint32_t n = 0;
    fn(physical, &n, nullptr);
    if (n == 0) return out;
    out.resize(n);
    for (auto& p : out) p.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    fn(physical, &n, out.data());
    out.resize(n);
    return out;
  }

  static bool has_device_ext(VkPhysicalDevice d, const char* name) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(d, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> e(n);
    vkEnumerateDeviceExtensionProperties(d, nullptr, &n, e.data());
    for (auto& x : e)
      if (std::strcmp(x.extensionName, name) == 0) return true;
    return false;
  }

  // ------------------------------------------------------------- buffers ----
  uint32_t find_mem_type(uint32_t mask, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i) {
      if ((mask & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & want) == want)
        return i;
    }
    return UINT32_MAX;
  }

  VkBuf create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, bool host_visible) {
    VkBuf b{};
    b.size = size;
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VKB_CHECK(vkCreateBuffer(device, &bci, nullptr, &b.buffer));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device, b.buffer, &req);
    VkMemoryPropertyFlags want = host_visible
                                     ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                                     : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    uint32_t mt = find_mem_type(req.memoryTypeBits, want);
    if (mt == UINT32_MAX && host_visible) mt = find_mem_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (mt == UINT32_MAX) {
      std::fprintf(stderr, "[vk] no memory type\n");
      std::exit(2);
    }
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mt;
    VKB_CHECK(vkAllocateMemory(device, &mai, nullptr, &b.memory));
    VKB_CHECK(vkBindBufferMemory(device, b.buffer, b.memory, 0));
    if (host_visible) VKB_CHECK(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));
    return b;
  }

  void destroy_buffer(VkBuf& b) {
    if (b.mapped) vkUnmapMemory(device, b.memory);
    if (b.buffer) vkDestroyBuffer(device, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(device, b.memory, nullptr);
    b = VkBuf{};
  }

  void begin_cmd(bool one_time = true) {
    VKB_CHECK(vkResetCommandBuffer(cmd, 0));
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = one_time ? VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT : 0;
    VKB_CHECK(vkBeginCommandBuffer(cmd, &bi));
  }

  void end_cmd_submit_wait() {
    VKB_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VKB_CHECK(vkQueueSubmit(queue, 1, &si, fence));
    VKB_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    VKB_CHECK(vkResetFences(device, 1, &fence));
  }

  void upload(VkBuf& dst, const void* src, VkDeviceSize size) {
    VkBuf stage = create_buffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
    std::memcpy(stage.mapped, src, (size_t)size);
    begin_cmd();
    VkBufferCopy c{};
    c.size = size;
    vkCmdCopyBuffer(cmd, stage.buffer, dst.buffer, 1, &c);
    VkBufferMemoryBarrier bar{};
    bar.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    bar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    bar.buffer = dst.buffer;
    bar.offset = 0;
    bar.size = size;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 0, nullptr, 1, &bar, 0, nullptr);
    end_cmd_submit_wait();
    destroy_buffer(stage);
  }

  void download(VkBuf& src, void* dst, VkDeviceSize size) {
    VkBuf stage = create_buffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    begin_cmd();
    VkBufferCopy c{};
    c.size = size;
    vkCmdCopyBuffer(cmd, src.buffer, stage.buffer, 1, &c);
    end_cmd_submit_wait();
    std::memcpy(dst, stage.mapped, (size_t)size);
    destroy_buffer(stage);
  }

  // ------------------------------------------------------------- shaders ----
  VkShaderModule load_shader(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) {
      std::fprintf(stderr, "[vk] cannot open %s\n", path);
      std::exit(2);
    }
    std::fseek(f, 0, SEEK_END);
    long len = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> code((size_t)len / 4);
    if (std::fread(code.data(), 1, (size_t)len, f) != (size_t)len) {
      std::fprintf(stderr, "[vk] short read %s\n", path);
      std::exit(2);
    }
    std::fclose(f);
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = (size_t)len;
    ci.pCode = code.data();
    VkShaderModule m = VK_NULL_HANDLE;
    VKB_CHECK(vkCreateShaderModule(device, &ci, nullptr, &m));
    return m;
  }

  VkDescriptorSetLayout make_set_layout(uint32_t n_storage) {
    std::vector<VkDescriptorSetLayoutBinding> b(n_storage);
    for (uint32_t i = 0; i < n_storage; ++i) {
      b[i].binding = i;
      b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      b[i].descriptorCount = 1;
      b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = (uint32_t)b.size();
    ci.pBindings = b.data();
    VkDescriptorSetLayout l = VK_NULL_HANDLE;
    VKB_CHECK(vkCreateDescriptorSetLayout(device, &ci, nullptr, &l));
    return l;
  }

  VkPipelineLayout make_pipeline_layout(VkDescriptorSetLayout set, uint32_t push_size) {
    VkPushConstantRange r{};
    r.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    r.offset = 0;
    r.size = push_size;
    VkPipelineLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    ci.setLayoutCount = 1;
    ci.pSetLayouts = &set;
    ci.pushConstantRangeCount = push_size ? 1 : 0;
    ci.pPushConstantRanges = &r;
    VkPipelineLayout l = VK_NULL_HANDLE;
    VKB_CHECK(vkCreatePipelineLayout(device, &ci, nullptr, &l));
    return l;
  }

  VkPipeline make_pipeline(VkShaderModule mod, VkPipelineLayout layout,
                           uint32_t required_subgroup_size) {
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo ssc{};
    ssc.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    ssc.requiredSubgroupSize = required_subgroup_size;
    VkComputePipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = mod;
    ci.stage.pName = "main";
    if (required_subgroup_size) ci.stage.pNext = &ssc;
    ci.layout = layout;
    VkPipeline p = VK_NULL_HANDLE;
    VkResult r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &p);
    if (r != VK_SUCCESS) {
      std::fprintf(stderr, "[vk] pipeline creation failed (%d) - variant not supported here\n", (int)r);
      return VK_NULL_HANDLE;
    }
    return p;
  }

  VkDescriptorSet alloc_set(VkDescriptorSetLayout layout, VkDescriptorPool pool) {
    VkDescriptorSetAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;
    VkDescriptorSet s = VK_NULL_HANDLE;
    VKB_CHECK(vkAllocateDescriptorSets(device, &ai, &s));
    return s;
  }

  void bind_buffers(VkDescriptorSet set, const std::vector<VkBuf*>& bufs) {
    std::vector<VkDescriptorBufferInfo> info(bufs.size());
    std::vector<VkWriteDescriptorSet> w(bufs.size());
    for (size_t i = 0; i < bufs.size(); ++i) {
      info[i].buffer = bufs[i]->buffer;
      info[i].offset = 0;
      info[i].range = VK_WHOLE_SIZE;
      w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[i].dstSet = set;
      w[i].dstBinding = (uint32_t)i;
      w[i].descriptorCount = 1;
      w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      w[i].pBufferInfo = &info[i];
    }
    vkUpdateDescriptorSets(device, (uint32_t)w.size(), w.data(), 0, nullptr);
  }

  // -------------------------------------------------------------- timing ----
  // Returns per-dispatch GPU time in ms (timestamps around each dispatch).
  std::vector<double> time_dispatches(VkPipeline pipe, VkPipelineLayout layout, VkDescriptorSet set,
                                      const void* push, uint32_t push_size, uint32_t gx, int iters,
                                      int warmup, uint32_t gy = 1, uint32_t gz = 1) {
    const int total = iters + warmup;
    VkQueryPoolCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qci.queryCount = (uint32_t)(total * 2);
    VkQueryPool qp = VK_NULL_HANDLE;
    VKB_CHECK(vkCreateQueryPool(device, &qci, nullptr, &qp));

    begin_cmd();
    vkCmdResetQueryPool(cmd, qp, 0, (uint32_t)(total * 2));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    if (push_size && push)
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);
    for (int i = 0; i < total; ++i) {
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, (uint32_t)(2 * i));
      vkCmdDispatch(cmd, gx, gy, gz);
      vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, (uint32_t)(2 * i + 1));
    }
    VKB_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VKB_CHECK(vkQueueSubmit(queue, 1, &si, fence));
    VKB_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    VKB_CHECK(vkResetFences(device, 1, &fence));

    std::vector<uint64_t> ts((size_t)total * 2);
    VKB_CHECK(vkGetQueryPoolResults(device, qp, 0, (uint32_t)ts.size(), ts.size() * sizeof(uint64_t),
                                    ts.data(), sizeof(uint64_t),
                                    VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    std::vector<double> out;
    out.reserve((size_t)iters);
    for (int i = warmup; i < total; ++i) {
      const double ns = (double)(ts[(size_t)2 * i + 1] - ts[(size_t)2 * i]) * (double)timestamp_period;
      out.push_back(ns * 1e-6);
    }
    vkDestroyQueryPool(device, qp, nullptr);
    return out;
  }
};
