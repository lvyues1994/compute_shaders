#include <tc/vulkan/vulkan_device.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "vk_handles.hpp"

namespace tc::vulkan {

namespace {

bool hasInstanceLayer(char const *name) {
    auto count = std::uint32_t{0};
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    auto layers = std::vector<VkLayerProperties>(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    return std::any_of(layers.begin(), layers.end(),
                       [name](VkLayerProperties const &layer) { return std::strcmp(layer.layerName, name) == 0; });
}

bool hasInstanceExtension(char const *name) {
    auto count = std::uint32_t{0};
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    auto extensions = std::vector<VkExtensionProperties>(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data());
    return std::any_of(extensions.begin(), extensions.end(), [name](VkExtensionProperties const &extension) {
        return std::strcmp(extension.extensionName, name) == 0;
    });
}

bool hasDeviceExtension(VkPhysicalDevice const physical, char const *name) {
    auto count = std::uint32_t{0};
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr);
    auto extensions = std::vector<VkExtensionProperties>(count);
    vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data());
    return std::any_of(extensions.begin(), extensions.end(), [name](VkExtensionProperties const &extension) {
        return std::strcmp(extension.extensionName, name) == 0;
    });
}

VkFormat vkFormatOf(ImageFormat const &format) {
    struct Entry {
        char const *qualifier;
        VkFormat format;
    };
    static constexpr Entry table[] = {
        {"r8", VK_FORMAT_R8_UNORM},
        {"rg8", VK_FORMAT_R8G8_UNORM},
        {"rgba8", VK_FORMAT_R8G8B8A8_UNORM},
        {"r8ui", VK_FORMAT_R8_UINT},
        {"rgba8ui", VK_FORMAT_R8G8B8A8_UINT},
        {"rgba8i", VK_FORMAT_R8G8B8A8_SINT},
        {"r32ui", VK_FORMAT_R32_UINT},
        {"rg32ui", VK_FORMAT_R32G32_UINT},
        {"rgba32ui", VK_FORMAT_R32G32B32A32_UINT},
        {"r32i", VK_FORMAT_R32_SINT},
        {"rgba32i", VK_FORMAT_R32G32B32A32_SINT},
        {"r32f", VK_FORMAT_R32_SFLOAT},
        {"rg32f", VK_FORMAT_R32G32_SFLOAT},
        {"rgba32f", VK_FORMAT_R32G32B32A32_SFLOAT},
    };
    auto const qualifier = glslFormatQualifier(format);
    for (auto const &entry : table) {
        if (qualifier == entry.qualifier) {
            return entry.format;
        }
    }
    throw DeviceError{"image format " + qualifier + " is not supported by the Vulkan backend"};
}

int deviceTypeRank(VkPhysicalDeviceType const type) {
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return 0;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return 1;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return 2;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return 3;
    default:
        return 4;
    }
}

std::optional<std::uint32_t> computeQueueFamily(VkPhysicalDevice const physical) {
    auto count = std::uint32_t{0};
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    auto families = std::vector<VkQueueFamilyProperties>(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    for (std::uint32_t i = 0; i < count; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
            return i;
        }
    }
    return std::nullopt;
}

// Makes every earlier write on the queue visible to every later access.
void fullBarrier(VkCommandBuffer const cmd) {
    auto barrier = vkStruct<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier,
                         0, nullptr, 0, nullptr);
}

void hostReadBarrier(VkCommandBuffer const cmd) {
    auto barrier = vkStruct<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr,
                         0, nullptr);
}

// Makes every write of the submission visible to the host, for buffers the
// CPU reads in place.
void hostVisibleBarrier(VkCommandBuffer const cmd) {
    auto barrier = vkStruct<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0,
                         nullptr, 0, nullptr);
}

// Instance and device shared by the VulkanDevice and every mapped buffer it
// hands out, so such buffers may outlive the device object.
struct VulkanContext {
    InstanceOwner instance;
    std::uint32_t apiVersion = VK_API_VERSION_1_0;
    bool externalImages = false; // AHardwareBuffer import with sampler YCbCr conversion
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    std::uint32_t queueFamily = 0;
    DeviceOwner device;
    VkQueue queue = VK_NULL_HANDLE;
};

// A buffer in host-visible device memory, persistently mapped.
struct MappedBuffer {
    std::shared_ptr<VulkanContext> context; // declared first: outlives the handles below
    DeviceObject<VkDeviceMemory> memory;
    DeviceObject<VkBuffer> buffer;
    std::byte *mapped = nullptr;
};

struct BufferMirror {
    std::weak_ptr<void const> owner;
    DeviceObject<VkDeviceMemory> memory; // declared before the buffer: freed after it
    DeviceObject<VkBuffer> buffer;
    std::shared_ptr<MappedBuffer> mapped; // set for createBuffer() allocations instead of the two above
    VkDeviceSize size = 0;

    VkBuffer handle() const noexcept { return mapped ? mapped->buffer.get() : buffer.get(); }
};

struct ImageMirror {
    std::weak_ptr<void const> owner;
    DeviceObject<VkDeviceMemory> memory;
    DeviceObject<VkImage> image;
    DeviceObject<VkImageView> view;
    ImageFormat format;
    Dim dim = Dim::D2;
    ivec3 extent{};
    VkImageUsageFlags usage = 0;
};

struct BoundTexture {
    void const *key = nullptr;
    Filter filter = Filter::Nearest;
    Wrap wrap = Wrap::ClampToEdge;
};

// A platform frame imported into Vulkan. The memory import holds its own
// reference to the hardware buffer.
struct ExternalMirror {
    std::weak_ptr<void const> owner;
    DeviceObject<VkSamplerYcbcrConversion> conversion;
    DeviceObject<VkDeviceMemory> memory;
    DeviceObject<VkImage> image;
    DeviceObject<VkImageView> view;
    DeviceObject<VkSampler> sampler;
};

template <class H>
std::uint64_t handleKey(H const handle) noexcept {
    if constexpr (std::is_pointer_v<H>) {
        return reinterpret_cast<std::uintptr_t>(handle);
    } else {
        return static_cast<std::uint64_t>(handle);
    }
}

// Takes a frame written outside Vulkan (camera, decoder, CPU) from the
// foreign queue family before sampling it.
void acquireForeignFrame(VkCommandBuffer const cmd, VkImage const image, std::uint32_t const queueFamily) {
    auto barrier = vkStruct<VkImageMemoryBarrier>(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER);
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_FOREIGN_EXT;
    barrier.dstQueueFamilyIndex = queueFamily;
    barrier.image = image;
    barrier.subresourceRange = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &barrier);
}

VkSamplerAddressMode addressMode(Wrap const wrap) {
    switch (wrap) {
    case Wrap::ClampToEdge:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case Wrap::Repeat:
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    case Wrap::MirroredRepeat:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    }
    return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
}

VkDescriptorType descriptorType(ResourceKind const kind) {
    switch (kind) {
    case ResourceKind::StorageBuffer:
        return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case ResourceKind::StorageImage:
        return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    case ResourceKind::SampledTexture:
    case ResourceKind::ExternalTexture:
        return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
}

struct PipelineObject {
    KernelProgram reflection;
    DeviceObject<VkShaderModule> module;
    DeviceObject<VkDescriptorSetLayout> setLayout;
    DeviceObject<VkPipelineLayout> layout;
    DeviceObject<VkPipeline> pipeline;
};

struct VulkanDevice final : Device {
    explicit VulkanDevice(VulkanDeviceConfig const &config);
    VulkanDevice(VulkanDevice const &) = delete;
    VulkanDevice &operator=(VulkanDevice const &) = delete;
    ~VulkanDevice() override {
        try {
            flush();
        } catch (DeviceError const &error) {
            std::fprintf(stderr, "[vulkan] queued work failed at shutdown: %s\n", error.what());
        }
    }

    DeviceInfo info() const override {
        auto const memory = sharedMemoryType();
        auto const deviceLocal = (memoryProperties.memoryTypes[memory].propertyFlags
                                  & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        return DeviceInfo{"vulkan", properties.deviceName, deviceLocal};
    }

protected:
    std::optional<ExternalMemory> doAllocateBuffer(ResourceIdentity const &identity, std::size_t bytes) override;
    void doUploadBuffer(ResourceIdentity const &identity, ConstByteSpan bytes) override;
    void doDownloadBuffer(ResourceIdentity const &identity, ByteSpan bytes) override;
    void doUploadImage(ImageTransfer const &transfer) override;
    void doDownloadImage(ImageTransfer const &transfer) override;
    void doUseKernel(KernelUse const &use) override;
    void doBindBuffer(BufferBindRequest const &request) override;
    void doBindImage(ImageBindRequest const &request) override;
    void doBindTexture(TextureBindRequest const &request) override;
    void doBindExternalTexture(ExternalTextureBindRequest const &request) override;
    void doSetUniform(UniformWrite const &write) override;
    void doExecute(Dispatch const &dispatch) override;
    void doFinish() override { flush(); }

private:
    VkFormatFeatureFlags formatFeatures(VkFormat format) const;
    VkSampler samplerFor(Filter filter, Wrap wrap);
    void createInstance(VulkanDeviceConfig const &config);
    void pickPhysicalDevice(VulkanDeviceConfig const &config);
    void createLogicalDevice();
    std::uint32_t memoryType(std::uint32_t typeBits, VkMemoryPropertyFlags required) const;
    std::uint32_t sharedMemoryType() const;
    DeviceObject<VkDeviceMemory> allocate(VkMemoryRequirements const &requirements, VkMemoryPropertyFlags flags);
    void ensureStaging(VkDeviceSize size);
    // Commands are batched in one command buffer until flush() submits it and
    // waits; recording() starts the batch on first use.
    VkCommandBuffer recording();
    void flush();
    VkDeviceSize stagingSlice(VkDeviceSize size);
    VkDescriptorSet allocateSet(VkDescriptorSetLayout layout);
    void collectGarbage();
    KernelProgram const &programFor(std::string_view name);
    PipelineObject &pipelineFor(KernelProgram const &program, std::vector<VkSampler> const &immutableSamplers);
    void acquireForeignFrame(VkCommandBuffer cmd, VkImage image) const {
        vulkan::acquireForeignFrame(cmd, image, queueFamily);
    }
#if defined(__ANDROID__)
    void importFrame(ExternalMirror &mirror, ExternalImage const &image);
#endif

    // Declared first: the context (instance and device) is destroyed after
    // every DeviceObject below and after the last mapped buffer.
    std::shared_ptr<VulkanContext> context = std::make_shared<VulkanContext>();
    InstanceOwner &instance = context->instance;
    VkPhysicalDevice &physical = context->physical;
    VkPhysicalDeviceProperties &properties = context->properties;
    VkPhysicalDeviceMemoryProperties &memoryProperties = context->memoryProperties;
    std::uint32_t &queueFamily = context->queueFamily;
    DeviceOwner &device = context->device;
    VkQueue &queue = context->queue;
    DeviceObject<VkCommandPool> commandPool;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    bool isRecording = false;
    DeviceObject<VkFence> fence;
    DeviceObject<VkDescriptorPool> descriptorPool;
    DeviceObject<VkDeviceMemory> stagingMemory;
    DeviceObject<VkBuffer> staging;
    std::byte *stagingMapped = nullptr;
    VkDeviceSize stagingCapacity = 0;
    VkDeviceSize stagingUsed = 0; // bytes handed out to the current batch
    std::unordered_map<void const *, BufferMirror> buffers;
    std::unordered_map<void const *, ImageMirror> images;
    std::unordered_map<void const *, ExternalMirror> externals;
    std::unordered_map<std::string, KernelProgram> programs;
    std::unordered_map<std::string, PipelineObject> pipelines;
    std::map<std::pair<Filter, Wrap>, DeviceObject<VkSampler>> samplers;

    KernelProgram const *current = nullptr;
    std::map<std::uint32_t, void const *> boundBuffers;
    std::map<std::uint32_t, void const *> boundImages;
    std::map<std::uint32_t, BoundTexture> boundTextures;
    std::map<std::uint32_t, void const *> boundExternals;
    std::map<std::uint32_t, bool> uniformsSet;
    std::array<std::byte, 256> pushConstants{};
};

VulkanDevice::VulkanDevice(VulkanDeviceConfig const &config) {
    createInstance(config);
    pickPhysicalDevice(config);
    createLogicalDevice();

    auto poolInfo = vkStruct<VkCommandPoolCreateInfo>(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamily;
    auto pool = VkCommandPool{};
    check(vkCreateCommandPool(device.handle, &poolInfo, nullptr, &pool), "vkCreateCommandPool");
    commandPool = DeviceObject<VkCommandPool>{device.handle, pool, vkDestroyCommandPool};

    auto allocInfo = vkStruct<VkCommandBufferAllocateInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
    allocInfo.commandPool = pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(device.handle, &allocInfo, &commandBuffer), "vkAllocateCommandBuffers");

    auto fenceInfo = vkStruct<VkFenceCreateInfo>(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    auto rawFence = VkFence{};
    check(vkCreateFence(device.handle, &fenceInfo, nullptr, &rawFence), "vkCreateFence");
    fence = DeviceObject<VkFence>{device.handle, rawFence, vkDestroyFence};

    VkDescriptorPoolSize const sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512},
                                          {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 512},
                                          {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 512}};
    auto descriptorInfo = vkStruct<VkDescriptorPoolCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
    descriptorInfo.maxSets = 128;
    descriptorInfo.poolSizeCount = 3;
    descriptorInfo.pPoolSizes = sizes;
    auto rawPool = VkDescriptorPool{};
    check(vkCreateDescriptorPool(device.handle, &descriptorInfo, nullptr, &rawPool), "vkCreateDescriptorPool");
    descriptorPool = DeviceObject<VkDescriptorPool>{device.handle, rawPool, vkDestroyDescriptorPool};
}

void VulkanDevice::createInstance(VulkanDeviceConfig const &config) {
    auto app = vkStruct<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
    app.pApplicationName = "tc";
    app.pEngineName = "tc";
    // Vulkan 1.1 when the loader has it: sampler YCbCr conversion (external
    // frames) is core there. Everything else needs only 1.0.
    auto const enumerateVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
        vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
    auto loaderVersion = std::uint32_t{VK_API_VERSION_1_0};
    if (enumerateVersion != nullptr) {
        enumerateVersion(&loaderVersion);
    }
    context->apiVersion = loaderVersion >= VK_API_VERSION_1_1 ? VK_API_VERSION_1_1 : VK_API_VERSION_1_0;
    app.apiVersion = context->apiVersion;
    auto extensions = std::vector<char const *>{};
    auto layers = std::vector<char const *>{};
    auto info = vkStruct<VkInstanceCreateInfo>(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
    if (hasInstanceExtension(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
    if (config.enableValidation) {
        if (hasInstanceLayer("VK_LAYER_KHRONOS_validation")) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
        } else {
            std::fprintf(stderr, "[vulkan] VK_LAYER_KHRONOS_validation is not installed; continuing without it\n");
        }
    }
    info.pApplicationInfo = &app;
    info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();
    info.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    info.ppEnabledLayerNames = layers.data();
    check(vkCreateInstance(&info, nullptr, &instance.handle), "vkCreateInstance");
}

void VulkanDevice::pickPhysicalDevice(VulkanDeviceConfig const &config) {
    auto count = std::uint32_t{0};
    check(vkEnumeratePhysicalDevices(instance.handle, &count, nullptr), "vkEnumeratePhysicalDevices");
    auto devices = std::vector<VkPhysicalDevice>(count);
    check(vkEnumeratePhysicalDevices(instance.handle, &count, devices.data()), "vkEnumeratePhysicalDevices");
    auto bestRank = 100;
    auto found = std::string{};
    for (auto const candidate : devices) {
        auto props = VkPhysicalDeviceProperties{};
        vkGetPhysicalDeviceProperties(candidate, &props);
        found += std::string{" '"} + props.deviceName + "'";
        auto const family = computeQueueFamily(candidate);
        auto const matches = config.deviceNameContains.empty()
                          or std::string{props.deviceName}.find(config.deviceNameContains) != std::string::npos;
        auto const rank = deviceTypeRank(props.deviceType);
        if (family and matches and rank < bestRank) {
            bestRank = rank;
            physical = candidate;
            properties = props;
            queueFamily = *family;
        }
    }
    if (physical == VK_NULL_HANDLE) {
        throw DeviceError{"no Vulkan device with a compute queue"
                          + (config.deviceNameContains.empty() ? std::string{}
                                                               : " matching '" + config.deviceNameContains + "'")
                          + (found.empty() ? std::string{} : " (found" + found + ")")};
    }
    vkGetPhysicalDeviceMemoryProperties(physical, &memoryProperties);
}

void VulkanDevice::createLogicalDevice() {
    auto supported = VkPhysicalDeviceFeatures{};
    vkGetPhysicalDeviceFeatures(physical, &supported);
    auto features = VkPhysicalDeviceFeatures{};
    features.shaderFloat64 = supported.shaderFloat64;
    features.shaderStorageImageExtendedFormats = supported.shaderStorageImageExtendedFormats;

    auto const priority = 1.0f;
    auto queueInfo = vkStruct<VkDeviceQueueCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    auto extensions = std::vector<char const *>{};
    if (hasDeviceExtension(physical, "VK_KHR_portability_subset")) {
        extensions.push_back("VK_KHR_portability_subset");
    }
    auto info = vkStruct<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
#if defined(__ANDROID__)
    auto ycbcr = vkStruct<VkPhysicalDeviceSamplerYcbcrConversionFeatures>(
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES);
    // Importing AHardwareBuffers (camera and decoder frames) needs Vulkan 1.1
    // for sampler YCbCr conversion plus two extensions.
    if (context->apiVersion >= VK_API_VERSION_1_1 and properties.apiVersion >= VK_API_VERSION_1_1
        and hasDeviceExtension(physical, VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME)
        and hasDeviceExtension(physical, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME)) {
        auto features2 = vkStruct<VkPhysicalDeviceFeatures2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
        features2.pNext = &ycbcr;
        vkGetPhysicalDeviceFeatures2(physical, &features2);
        if (ycbcr.samplerYcbcrConversion == VK_TRUE) {
            extensions.push_back(VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME);
            extensions.push_back(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
            ycbcr.pNext = nullptr;
            info.pNext = &ycbcr;
            context->externalImages = true;
        }
    }
#endif
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queueInfo;
    info.pEnabledFeatures = &features;
    info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();
    check(vkCreateDevice(physical, &info, nullptr, &device.handle), "vkCreateDevice");
    vkGetDeviceQueue(device.handle, queueFamily, 0, &queue);
}

std::uint32_t VulkanDevice::memoryType(std::uint32_t const typeBits, VkMemoryPropertyFlags const required) const {
    for (std::uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) != 0 and (memoryProperties.memoryTypes[i].propertyFlags & required) == required) {
            return i;
        }
    }
    throw DeviceError{"no Vulkan memory type with the required properties"};
}

// Memory for createBuffer(): CPU-visible and coherent. Device-local memory
// wins when its heap is large (unified memory, resizable BAR) rather than the
// small BAR window of a discrete GPU; otherwise cached system memory.
std::uint32_t VulkanDevice::sharedMemoryType() const {
    auto constexpr hostShared = VkMemoryPropertyFlags{VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                      | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
    auto largestDeviceHeap = VkDeviceSize{0};
    for (std::uint32_t h = 0; h < memoryProperties.memoryHeapCount; ++h) {
        if ((memoryProperties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            largestDeviceHeap = std::max(largestDeviceHeap, memoryProperties.memoryHeaps[h].size);
        }
    }
    auto best = std::uint32_t{0};
    auto bestScore = -100;
    for (std::uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        auto const &type = memoryProperties.memoryTypes[i];
        if ((type.propertyFlags & hostShared) != hostShared) {
            continue;
        }
        auto const heapSize = memoryProperties.memoryHeaps[type.heapIndex].size;
        auto const deviceLocal = (type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        auto const largeHeap = heapSize * 2 >= largestDeviceHeap;
        auto score = deviceLocal ? (largeHeap ? 4 : -1) : 0;
        score += (type.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0 ? 2 : 0;
        if (score > bestScore) {
            best = i;
            bestScore = score;
        }
    }
    if (bestScore == -100) {
        throw DeviceError{"no host-visible coherent Vulkan memory"};
    }
    return best;
}

DeviceObject<VkDeviceMemory> VulkanDevice::allocate(VkMemoryRequirements const &requirements,
                                                    VkMemoryPropertyFlags const flags) {
    auto info = vkStruct<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    info.allocationSize = requirements.size;
    info.memoryTypeIndex = memoryType(requirements.memoryTypeBits, flags);
    auto memory = VkDeviceMemory{};
    check(vkAllocateMemory(device.handle, &info, nullptr, &memory), "vkAllocateMemory");
    return DeviceObject<VkDeviceMemory>{device.handle, memory, vkFreeMemory};
}

std::optional<ExternalMemory> VulkanDevice::doAllocateBuffer(ResourceIdentity const &identity,
                                                             std::size_t const bytes) {
    if (bytes == 0) {
        return std::nullopt;
    }
    collectGarbage();
    auto allocation = std::make_shared<MappedBuffer>();
    allocation->context = context;
    auto info = vkStruct<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    info.size = bytes;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto buffer = VkBuffer{};
    check(vkCreateBuffer(device.handle, &info, nullptr, &buffer), "vkCreateBuffer");
    allocation->buffer = DeviceObject<VkBuffer>{device.handle, buffer, vkDestroyBuffer};
    auto requirements = VkMemoryRequirements{};
    vkGetBufferMemoryRequirements(device.handle, buffer, &requirements);
    auto type = sharedMemoryType();
    if ((requirements.memoryTypeBits & (1u << type)) == 0) {
        type = memoryType(requirements.memoryTypeBits,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    }
    auto allocateInfo = vkStruct<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    allocateInfo.allocationSize = requirements.size;
    allocateInfo.memoryTypeIndex = type;
    auto memory = VkDeviceMemory{};
    check(vkAllocateMemory(device.handle, &allocateInfo, nullptr, &memory), "vkAllocateMemory");
    allocation->memory = DeviceObject<VkDeviceMemory>{device.handle, memory, vkFreeMemory};
    check(vkBindBufferMemory(device.handle, buffer, memory, 0), "vkBindBufferMemory");
    void *mapped = nullptr;
    check(vkMapMemory(device.handle, memory, 0, VK_WHOLE_SIZE, 0, &mapped), "vkMapMemory");
    allocation->mapped = static_cast<std::byte *>(mapped);
    std::memset(allocation->mapped, 0, bytes);

    auto &mirror = buffers[identity.key()];
    mirror = BufferMirror{};
    mirror.owner = identity.watch();
    mirror.mapped = allocation;
    mirror.size = bytes;
    return ExternalMemory{allocation->mapped, bytes, allocation};
}

void VulkanDevice::ensureStaging(VkDeviceSize const size) {
    if (stagingCapacity >= size) {
        return;
    }
    auto const capacity = std::max<VkDeviceSize>(size, 1u << 20);
    staging = DeviceObject<VkBuffer>{};
    stagingMemory = DeviceObject<VkDeviceMemory>{};
    auto info = vkStruct<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    info.size = capacity;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto buffer = VkBuffer{};
    check(vkCreateBuffer(device.handle, &info, nullptr, &buffer), "vkCreateBuffer");
    staging = DeviceObject<VkBuffer>{device.handle, buffer, vkDestroyBuffer};
    auto requirements = VkMemoryRequirements{};
    vkGetBufferMemoryRequirements(device.handle, buffer, &requirements);
    stagingMemory = allocate(requirements, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check(vkBindBufferMemory(device.handle, buffer, stagingMemory.get(), 0), "vkBindBufferMemory");
    void *mapped = nullptr;
    check(vkMapMemory(device.handle, stagingMemory.get(), 0, capacity, 0, &mapped), "vkMapMemory");
    stagingMapped = static_cast<std::byte *>(mapped);
    stagingCapacity = capacity;
}

// The batch's command buffer, begun on first use. Every command is preceded
// by a full barrier, so it waits for and sees all earlier commands.
VkCommandBuffer VulkanDevice::recording() {
    if (not isRecording) {
        check(vkResetCommandBuffer(commandBuffer, 0), "vkResetCommandBuffer");
        auto begin = vkStruct<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(commandBuffer, &begin), "vkBeginCommandBuffer");
        isRecording = true;
    }
    fullBarrier(commandBuffer);
    return commandBuffer;
}

void VulkanDevice::flush() {
    if (not isRecording) {
        return;
    }
    isRecording = false;
    hostVisibleBarrier(commandBuffer);
    check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
    auto info = vkStruct<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
    info.commandBufferCount = 1;
    info.pCommandBuffers = &commandBuffer;
    auto const rawFence = fence.get();
    check(vkQueueSubmit(queue, 1, &info, rawFence), "vkQueueSubmit");
    check(vkWaitForFences(device.handle, 1, &rawFence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    check(vkResetFences(device.handle, 1, &rawFence), "vkResetFences");
    stagingUsed = 0;
    check(vkResetDescriptorPool(device.handle, descriptorPool.get(), 0), "vkResetDescriptorPool");
    collectGarbage();
}

// Offset of `size` staging bytes no other command of this batch uses.
VkDeviceSize VulkanDevice::stagingSlice(VkDeviceSize const size) {
    auto const offset = (stagingUsed + 15) / 16 * 16;
    if (offset + size <= stagingCapacity) {
        stagingUsed = offset + size;
        return offset;
    }
    flush();
    ensureStaging(size);
    stagingUsed = size;
    return 0;
}

VkDescriptorSet VulkanDevice::allocateSet(VkDescriptorSetLayout const layout) {
    auto info = vkStruct<VkDescriptorSetAllocateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
    info.descriptorPool = descriptorPool.get();
    info.descriptorSetCount = 1;
    info.pSetLayouts = &layout;
    auto set = VkDescriptorSet{};
    auto const result = vkAllocateDescriptorSets(device.handle, &info, &set);
    if (result == VK_ERROR_OUT_OF_POOL_MEMORY or result == VK_ERROR_FRAGMENTED_POOL) {
        flush(); // resets the pool
        check(vkAllocateDescriptorSets(device.handle, &info, &set), "vkAllocateDescriptorSets");
    } else {
        check(result, "vkAllocateDescriptorSets");
    }
    return set;
}

// Releases mirrors of destroyed host resources; only safe with no queued work.
void VulkanDevice::collectGarbage() {
    if (isRecording) {
        return;
    }
    for (auto it = buffers.begin(); it != buffers.end();) {
        it = it->second.owner.expired() ? buffers.erase(it) : std::next(it);
    }
    for (auto it = images.begin(); it != images.end();) {
        it = it->second.owner.expired() ? images.erase(it) : std::next(it);
    }
    for (auto it = externals.begin(); it != externals.end();) {
        if (not it->second.owner.expired()) {
            ++it;
            continue;
        }
        // Pipelines bake the frame's sampler into their layout; drop them too.
        auto const token = "|" + std::to_string(handleKey(it->second.sampler.get()));
        for (auto pipeline = pipelines.begin(); pipeline != pipelines.end();) {
            pipeline = pipeline->first.find(token) != std::string::npos ? pipelines.erase(pipeline) : std::next(pipeline);
        }
        it = externals.erase(it);
    }
}

void VulkanDevice::doUploadBuffer(ResourceIdentity const &identity, ConstByteSpan const bytes) {
    if (bytes.empty()) {
        throw DeviceError{"uploadBuffer: Vulkan buffers cannot be empty"};
    }
    collectGarbage();
    auto &mirror = buffers[identity.key()];
    if (mirror.mapped and mirror.size == bytes.size() and mirror.mapped->mapped == bytes.data()) {
        return; // createBuffer() memory: the device reads it in place
    }
    if (mirror.size != bytes.size()) {
        if (mirror.size != 0) {
            flush(); // queued commands may still use the old buffer
        }
        mirror = BufferMirror{};
        mirror.owner = identity.watch();
        auto info = vkStruct<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
        info.size = bytes.size();
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                   | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        auto buffer = VkBuffer{};
        check(vkCreateBuffer(device.handle, &info, nullptr, &buffer), "vkCreateBuffer");
        mirror.buffer = DeviceObject<VkBuffer>{device.handle, buffer, vkDestroyBuffer};
        auto requirements = VkMemoryRequirements{};
        vkGetBufferMemoryRequirements(device.handle, buffer, &requirements);
        mirror.memory = allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkBindBufferMemory(device.handle, buffer, mirror.memory.get(), 0), "vkBindBufferMemory");
        mirror.size = bytes.size();
    }
    auto const offset = stagingSlice(bytes.size());
    std::memcpy(stagingMapped + offset, bytes.data(), bytes.size());
    auto const region = VkBufferCopy{offset, 0, bytes.size()};
    vkCmdCopyBuffer(recording(), staging.get(), mirror.handle(), 1, &region);
}

void VulkanDevice::doDownloadBuffer(ResourceIdentity const &identity, ByteSpan const bytes) {
    auto const it = buffers.find(identity.key());
    if (it == buffers.end() or it->second.size != bytes.size()) {
        throw DeviceError{"downloadBuffer: the buffer was never uploaded to this device (or changed size)"};
    }
    if (it->second.mapped and it->second.mapped->mapped == bytes.data()) {
        flush(); // createBuffer() memory: GPU writes are visible once the batch completes
        return;
    }
    auto const offset = stagingSlice(bytes.size());
    auto const region = VkBufferCopy{0, offset, bytes.size()};
    auto const cmd = recording();
    vkCmdCopyBuffer(cmd, it->second.handle(), staging.get(), 1, &region);
    hostReadBarrier(cmd);
    flush();
    std::memcpy(bytes.data(), stagingMapped + offset, bytes.size());
}

void VulkanDevice::doUploadImage(ImageTransfer const &transfer) {
    collectGarbage();
    auto const format = vkFormatOf(transfer.format);
    auto &mirror = images[transfer.identity->key()];
    auto const isNew = mirror.image.get() == VK_NULL_HANDLE or not(mirror.format == transfer.format)
                    or not(mirror.extent == transfer.extent);
    if (isNew) {
        // The same image may be written as a storage image and read as a
        // sampled texture later, so request every usage the format supports.
        auto const features = formatFeatures(format);
        auto usage = VkImageUsageFlags{VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        if ((features & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0) {
            usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        }
        if ((features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0) {
            usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
        }
        if ((usage & (VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) == 0) {
            throw DeviceError{"image format " + glslFormatQualifier(transfer.format)
                              + " can be neither stored nor sampled on " + properties.deviceName};
        }
        if (mirror.image.get() != VK_NULL_HANDLE) {
            flush(); // queued commands may still use the old image
        }
        mirror = ImageMirror{};
        mirror.usage = usage;
        mirror.owner = transfer.identity->watch();
        mirror.format = transfer.format;
        mirror.dim = transfer.dim;
        mirror.extent = transfer.extent;
        auto info = vkStruct<VkImageCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);
        info.imageType = transfer.dim == Dim::D3 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = VkExtent3D{static_cast<std::uint32_t>(transfer.extent.x),
                                 static_cast<std::uint32_t>(transfer.extent.y),
                                 static_cast<std::uint32_t>(transfer.extent.z)};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = usage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        auto image = VkImage{};
        check(vkCreateImage(device.handle, &info, nullptr, &image), "vkCreateImage");
        mirror.image = DeviceObject<VkImage>{device.handle, image, vkDestroyImage};
        auto requirements = VkMemoryRequirements{};
        vkGetImageMemoryRequirements(device.handle, image, &requirements);
        mirror.memory = allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkBindImageMemory(device.handle, image, mirror.memory.get(), 0), "vkBindImageMemory");
        auto viewInfo = vkStruct<VkImageViewCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
        viewInfo.image = image;
        viewInfo.viewType = transfer.dim == Dim::D3 ? VK_IMAGE_VIEW_TYPE_3D : VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        auto view = VkImageView{};
        check(vkCreateImageView(device.handle, &viewInfo, nullptr, &view), "vkCreateImageView");
        mirror.view = DeviceObject<VkImageView>{device.handle, view, vkDestroyImageView};
    }
    auto const texelCount = static_cast<std::size_t>(transfer.extent.x) * transfer.extent.y * transfer.extent.z;
    auto const size = texelCount * transfer.format.bytesPerTexel();
    auto const offset = stagingSlice(size);
    transfer.writeTexels(ByteSpan{stagingMapped + offset, size});
    auto const image = mirror.image.get();
    auto const cmd = recording();
    if (isNew) {
        auto barrier = vkStruct<VkImageMemoryBarrier>(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER);
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                             nullptr, 1, &barrier);
    }
    auto region = VkBufferImageCopy{};
    region.bufferOffset = offset;
    region.imageSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = VkExtent3D{static_cast<std::uint32_t>(transfer.extent.x),
                                    static_cast<std::uint32_t>(transfer.extent.y),
                                    static_cast<std::uint32_t>(transfer.extent.z)};
    vkCmdCopyBufferToImage(cmd, staging.get(), image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
}

void VulkanDevice::doDownloadImage(ImageTransfer const &transfer) {
    auto const it = images.find(transfer.identity->key());
    if (it == images.end() or not(it->second.format == transfer.format) or not(it->second.extent == transfer.extent)) {
        throw DeviceError{"downloadImage: the image was never uploaded to this device"};
    }
    auto const texelCount = static_cast<std::size_t>(transfer.extent.x) * transfer.extent.y * transfer.extent.z;
    auto const size = texelCount * transfer.format.bytesPerTexel();
    auto const offset = stagingSlice(size);
    auto region = VkBufferImageCopy{};
    region.bufferOffset = offset;
    region.imageSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = VkExtent3D{static_cast<std::uint32_t>(transfer.extent.x),
                                    static_cast<std::uint32_t>(transfer.extent.y),
                                    static_cast<std::uint32_t>(transfer.extent.z)};
    auto const cmd = recording();
    vkCmdCopyImageToBuffer(cmd, it->second.image.get(), VK_IMAGE_LAYOUT_GENERAL, staging.get(), 1, &region);
    hostReadBarrier(cmd);
    flush();
    transfer.readTexels(ConstByteSpan{stagingMapped + offset, size});
}

KernelProgram const &VulkanDevice::programFor(std::string_view const name) {
    auto const key = std::string{name};
    if (auto const it = programs.find(key); it != programs.end()) {
        return it->second;
    }
    auto const reflection = ProgramRegistry::instance().find(name);
    if (not reflection) {
        throw DeviceError{"kernel '" + key + "' was not transpiled; add its header with tc_add_kernels()"};
    }
    if (reflection->spirv.empty()) {
        throw DeviceError{"kernel '" + key + "' has no SPIR-V"};
    }
    if (reflection->pushConstantSize > properties.limits.maxPushConstantsSize) {
        throw DeviceError{"kernel '" + key + "' needs more push-constant space than " + properties.deviceName
                          + " provides"};
    }
    return programs.emplace(key, *reflection).first->second;
}

// One pipeline per kernel and set of immutable samplers: external frames
// need their YCbCr-converting sampler baked into the descriptor set layout.
PipelineObject &VulkanDevice::pipelineFor(KernelProgram const &program, std::vector<VkSampler> const &immutableSamplers) {
    auto key = std::string{program.name};
    for (auto const sampler : immutableSamplers) {
        key += "|" + std::to_string(handleKey(sampler));
    }
    if (auto const it = pipelines.find(key); it != pipelines.end()) {
        return it->second;
    }
    auto pipeline = PipelineObject{};
    pipeline.reflection = program;

    auto moduleInfo = vkStruct<VkShaderModuleCreateInfo>(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
    moduleInfo.codeSize = program.spirv.size() * sizeof(std::uint32_t);
    moduleInfo.pCode = program.spirv.data();
    auto module = VkShaderModule{};
    check(vkCreateShaderModule(device.handle, &moduleInfo, nullptr, &module), "vkCreateShaderModule");
    pipeline.module = DeviceObject<VkShaderModule>{device.handle, module, vkDestroyShaderModule};

    auto bindings = std::vector<VkDescriptorSetLayoutBinding>{};
    for (std::size_t i = 0; i < program.resources.size(); ++i) {
        auto const &slot = program.resources[i];
        auto binding = VkDescriptorSetLayoutBinding{};
        binding.binding = slot.binding;
        binding.descriptorType = descriptorType(slot.kind);
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        binding.pImmutableSamplers = immutableSamplers[i] != VK_NULL_HANDLE ? &immutableSamplers[i] : nullptr;
        bindings.push_back(binding);
    }
    auto setInfo = vkStruct<VkDescriptorSetLayoutCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
    setInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    setInfo.pBindings = bindings.data();
    auto setLayout = VkDescriptorSetLayout{};
    check(vkCreateDescriptorSetLayout(device.handle, &setInfo, nullptr, &setLayout), "vkCreateDescriptorSetLayout");
    pipeline.setLayout = DeviceObject<VkDescriptorSetLayout>{device.handle, setLayout, vkDestroyDescriptorSetLayout};

    auto const range = VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, program.pushConstantSize};
    auto layoutInfo = vkStruct<VkPipelineLayoutCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &setLayout;
    layoutInfo.pushConstantRangeCount = program.pushConstantSize > 0 ? 1 : 0;
    layoutInfo.pPushConstantRanges = &range;
    auto layout = VkPipelineLayout{};
    check(vkCreatePipelineLayout(device.handle, &layoutInfo, nullptr, &layout), "vkCreatePipelineLayout");
    pipeline.layout = DeviceObject<VkPipelineLayout>{device.handle, layout, vkDestroyPipelineLayout};

    auto info = vkStruct<VkComputePipelineCreateInfo>(VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO);
    info.stage = vkStruct<VkPipelineShaderStageCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.layout = layout;
    auto raw = VkPipeline{};
    check(vkCreateComputePipelines(device.handle, VK_NULL_HANDLE, 1, &info, nullptr, &raw), "vkCreateComputePipelines");
    pipeline.pipeline = DeviceObject<VkPipeline>{device.handle, raw, vkDestroyPipeline};
    return pipelines.emplace(key, std::move(pipeline)).first->second;
}

void VulkanDevice::doUseKernel(KernelUse const &use) {
    auto const &program = programFor(use.programName);
    auto const &size = program.localSize;
    if (not(size == use.localSize)) {
        throw DeviceError{"kernel '" + std::string{use.programName}
                          + "': local_size differs from the transpiled value; re-run the build"};
    }
    auto const &limits = properties.limits;
    if (size.x * size.y * size.z > limits.maxComputeWorkGroupInvocations or size.x > limits.maxComputeWorkGroupSize[0]
        or size.y > limits.maxComputeWorkGroupSize[1] or size.z > limits.maxComputeWorkGroupSize[2]) {
        throw DeviceError{"kernel '" + std::string{use.programName} + "' exceeds the workgroup size limits of "
                          + properties.deviceName};
    }
    current = &program;
    boundBuffers.clear();
    boundImages.clear();
    boundTextures.clear();
    boundExternals.clear();
    uniformsSet.clear();
}

void VulkanDevice::doBindBuffer(BufferBindRequest const &request) {
    if (buffers.count(request.identity->key()) == 0) {
        throw DeviceError{"bindBuffer: binding " + std::to_string(request.binding)
                          + " refers to a buffer that was never uploaded"};
    }
    boundBuffers[request.binding] = request.identity->key();
}

void VulkanDevice::doBindImage(ImageBindRequest const &request) {
    auto const it = images.find(request.identity->key());
    if (it == images.end() or not(it->second.format == request.format)) {
        throw DeviceError{"bindImage: binding " + std::to_string(request.binding)
                          + " refers to an image that was never uploaded"};
    }
    if ((it->second.usage & VK_IMAGE_USAGE_STORAGE_BIT) == 0) {
        throw DeviceError{"bindImage: format " + glslFormatQualifier(request.format)
                          + " cannot be a storage image on " + properties.deviceName + "; read it as a texture"};
    }
    boundImages[request.binding] = request.identity->key();
}

void VulkanDevice::doBindTexture(TextureBindRequest const &request) {
    auto const it = images.find(request.identity->key());
    if (it == images.end() or not(it->second.format == request.format)) {
        throw DeviceError{"bindTexture: binding " + std::to_string(request.binding)
                          + " refers to a texture that was never uploaded"};
    }
    if ((it->second.usage & VK_IMAGE_USAGE_SAMPLED_BIT) == 0) {
        throw DeviceError{"bindTexture: format " + glslFormatQualifier(request.format) + " cannot be sampled on "
                          + properties.deviceName};
    }
    auto const linearOk = (formatFeatures(vkFormatOf(request.format)) & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
    if (request.filter == Filter::Linear and not linearOk) {
        throw DeviceError{"bindTexture: format " + glslFormatQualifier(request.format)
                          + " cannot be filtered linearly on " + properties.deviceName};
    }
    boundTextures[request.binding] = BoundTexture{request.identity->key(), request.filter, request.wrap};
}

void VulkanDevice::doBindExternalTexture(ExternalTextureBindRequest const &request) {
#if defined(__ANDROID__)
    if (not context->externalImages) {
        throw DeviceError{"bindExternalTexture: " + std::string{properties.deviceName}
                          + " cannot import hardware buffers (needs Vulkan 1.1 and "
                            "VK_ANDROID_external_memory_android_hardware_buffer)"};
    }
    collectGarbage();
    auto const &identity = request.image->identity();
    auto &mirror = externals[identity.key()];
    if (mirror.image.get() == VK_NULL_HANDLE) {
        mirror.owner = identity.watch();
        importFrame(mirror, *request.image);
    }
    boundExternals[request.binding] = identity.key();
#else
    (void)request;
    throw DeviceError{"external images are only available on Android"};
#endif
}

#if defined(__ANDROID__)
// Imports an AHardwareBuffer as a sampled image. YUV frames, including
// vendor formats Vulkan only knows as an "external format", are converted to
// RGB by a sampler YCbCr conversion (BT.601 limited range, as on the CPU).
void VulkanDevice::importFrame(ExternalMirror &mirror, ExternalImage const &image) {
    auto *const buffer = image.hardwareBuffer();
    auto const getProperties = reinterpret_cast<PFN_vkGetAndroidHardwareBufferPropertiesANDROID>(
        vkGetDeviceProcAddr(device.handle, "vkGetAndroidHardwareBufferPropertiesANDROID"));
    auto formatProperties = vkStruct<VkAndroidHardwareBufferFormatPropertiesANDROID>(
        VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID);
    auto bufferProperties =
        vkStruct<VkAndroidHardwareBufferPropertiesANDROID>(VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID);
    bufferProperties.pNext = &formatProperties;
    check(getProperties(device.handle, buffer, &bufferProperties), "vkGetAndroidHardwareBufferPropertiesANDROID");

    auto const format = formatProperties.format;
    auto const isRgba = format == VK_FORMAT_R8G8B8A8_UNORM or format == VK_FORMAT_R8G8B8_UNORM;
    auto externalFormat = vkStruct<VkExternalFormatANDROID>(VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID);
    externalFormat.externalFormat = format == VK_FORMAT_UNDEFINED ? formatProperties.externalFormat : 0;

    auto const linear = isRgba
                     or (formatProperties.formatFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT)
                            != 0;
    auto const filter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    auto conversionInfo = vkStruct<VkSamplerYcbcrConversionInfo>(VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO);
    if (not isRgba) {
        auto info = vkStruct<VkSamplerYcbcrConversionCreateInfo>(VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO);
        info.pNext = &externalFormat;
        info.format = format;
        info.ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_601;
        info.ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_NARROW;
        info.components = formatProperties.samplerYcbcrConversionComponents;
        info.xChromaOffset = formatProperties.suggestedXChromaOffset;
        info.yChromaOffset = formatProperties.suggestedYChromaOffset;
        info.chromaFilter = filter;
        auto conversion = VkSamplerYcbcrConversion{};
        check(vkCreateSamplerYcbcrConversion(device.handle, &info, nullptr, &conversion),
              "vkCreateSamplerYcbcrConversion");
        mirror.conversion =
            DeviceObject<VkSamplerYcbcrConversion>{device.handle, conversion, vkDestroySamplerYcbcrConversion};
        conversionInfo.conversion = conversion;
    }

    auto desc = AHardwareBuffer_Desc{};
    AHardwareBuffer_describe(buffer, &desc);
    auto externalMemory = vkStruct<VkExternalMemoryImageCreateInfo>(VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO);
    externalMemory.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
    externalMemory.pNext = &externalFormat;
    auto imageInfo = vkStruct<VkImageCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);
    imageInfo.pNext = &externalMemory;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = format;
    imageInfo.extent = VkExtent3D{desc.width, desc.height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    auto rawImage = VkImage{};
    check(vkCreateImage(device.handle, &imageInfo, nullptr, &rawImage), "vkCreateImage (hardware buffer)");
    mirror.image = DeviceObject<VkImage>{device.handle, rawImage, vkDestroyImage};

    auto dedicated = vkStruct<VkMemoryDedicatedAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO);
    dedicated.image = rawImage;
    auto import = vkStruct<VkImportAndroidHardwareBufferInfoANDROID>(
        VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID);
    import.pNext = &dedicated;
    import.buffer = buffer;
    auto allocateInfo = vkStruct<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    allocateInfo.pNext = &import;
    allocateInfo.allocationSize = bufferProperties.allocationSize;
    allocateInfo.memoryTypeIndex = memoryType(bufferProperties.memoryTypeBits, 0);
    auto memory = VkDeviceMemory{};
    check(vkAllocateMemory(device.handle, &allocateInfo, nullptr, &memory), "vkAllocateMemory (hardware buffer)");
    mirror.memory = DeviceObject<VkDeviceMemory>{device.handle, memory, vkFreeMemory};
    check(vkBindImageMemory(device.handle, rawImage, memory, 0), "vkBindImageMemory (hardware buffer)");

    auto viewInfo = vkStruct<VkImageViewCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
    viewInfo.pNext = isRgba ? nullptr : &conversionInfo;
    viewInfo.image = rawImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    auto view = VkImageView{};
    check(vkCreateImageView(device.handle, &viewInfo, nullptr, &view), "vkCreateImageView (hardware buffer)");
    mirror.view = DeviceObject<VkImageView>{device.handle, view, vkDestroyImageView};

    auto samplerInfo = vkStruct<VkSamplerCreateInfo>(VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO);
    samplerInfo.pNext = isRgba ? nullptr : &conversionInfo;
    samplerInfo.magFilter = filter;
    samplerInfo.minFilter = filter;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    auto sampler = VkSampler{};
    check(vkCreateSampler(device.handle, &samplerInfo, nullptr, &sampler), "vkCreateSampler (hardware buffer)");
    mirror.sampler = DeviceObject<VkSampler>{device.handle, sampler, vkDestroySampler};
}
#endif

VkFormatFeatureFlags VulkanDevice::formatFeatures(VkFormat const format) const {
    auto formatProperties = VkFormatProperties{};
    vkGetPhysicalDeviceFormatProperties(physical, format, &formatProperties);
    return formatProperties.optimalTilingFeatures;
}

VkSampler VulkanDevice::samplerFor(Filter const filter, Wrap const wrap) {
    auto const key = std::make_pair(filter, wrap);
    auto it = samplers.find(key);
    if (it == samplers.end()) {
        auto info = vkStruct<VkSamplerCreateInfo>(VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO);
        info.magFilter = filter == Filter::Linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        info.minFilter = info.magFilter;
        info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        info.addressModeU = addressMode(wrap);
        info.addressModeV = addressMode(wrap);
        info.addressModeW = addressMode(wrap);
        info.maxLod = 0.0f;
        auto sampler = VkSampler{};
        check(vkCreateSampler(device.handle, &info, nullptr, &sampler), "vkCreateSampler");
        it = samplers.emplace(key, DeviceObject<VkSampler>{device.handle, sampler, vkDestroySampler}).first;
    }
    return it->second.get();
}

void VulkanDevice::doSetUniform(UniformWrite const &write) {
    if (current == nullptr) {
        throw DeviceError{"setUniform: call useKernel() first"};
    }
    for (auto const &slot : current->uniforms) {
        if (slot.location == write.location) {
            std::memcpy(pushConstants.data() + slot.pushConstantOffset, write.bytes.data(), write.bytes.size());
            uniformsSet[slot.location] = true;
            return;
        }
    }
    throw DeviceError{"setUniform: kernel '" + std::string{current->name} + "' has no uniform at location "
                      + std::to_string(write.location)};
}

void VulkanDevice::doExecute(Dispatch const &dispatch) {
    if (current == nullptr or current->name != dispatch.programName) {
        throw DeviceError{"execute: call useKernel() for '" + std::string{dispatch.programName} + "' first"};
    }
    auto const &reflection = *current;
    auto const &limits = properties.limits;
    for (int axis = 0; axis < 3; ++axis) {
        if (dispatch.workGroups[axis] > limits.maxComputeWorkGroupCount[axis]) {
            throw DeviceError{"workgroup count exceeds the device limit"};
        }
    }
    for (auto const &slot : reflection.uniforms) {
        if (uniformsSet.count(slot.location) == 0) {
            throw DeviceError{"execute: uniform '" + std::string{slot.name} + "' was never set with setUniform()"};
        }
    }
    auto immutableSamplers = std::vector<VkSampler>(reflection.resources.size(), VK_NULL_HANDLE);
    auto frames = std::vector<VkImage>{};
    for (std::size_t i = 0; i < reflection.resources.size(); ++i) {
        auto const &slot = reflection.resources[i];
        if (slot.kind != ResourceKind::ExternalTexture) {
            continue;
        }
        auto const bound = boundExternals.find(slot.binding);
        if (bound == boundExternals.end()) {
            throw DeviceError{"execute: external texture '" + std::string{slot.name} + "' is not bound"};
        }
        auto const &frame = externals.at(bound->second);
        immutableSamplers[i] = frame.sampler.get();
        frames.push_back(frame.image.get());
    }
    auto &pipeline = pipelineFor(reflection, immutableSamplers);
    auto const set = allocateSet(pipeline.setLayout.get());

    auto bufferInfos = std::vector<VkDescriptorBufferInfo>{};
    auto imageInfos = std::vector<VkDescriptorImageInfo>{};
    bufferInfos.reserve(reflection.resources.size());
    imageInfos.reserve(reflection.resources.size());
    auto writes = std::vector<VkWriteDescriptorSet>{};
    for (auto const &slot : reflection.resources) {
        auto const name = std::string{slot.name} + "' (binding " + std::to_string(slot.binding) + ")";
        auto write = vkStruct<VkWriteDescriptorSet>(VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET);
        write.dstSet = set;
        write.dstBinding = slot.binding;
        write.descriptorCount = 1;
        if (slot.kind == ResourceKind::StorageBuffer) {
            auto const bound = boundBuffers.find(slot.binding);
            if (bound == boundBuffers.end()) {
                throw DeviceError{"execute: buffer '" + name + " is not bound"};
            }
            bufferInfos.push_back(VkDescriptorBufferInfo{buffers.at(bound->second).handle(), 0, VK_WHOLE_SIZE});
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &bufferInfos.back();
        } else if (slot.kind == ResourceKind::SampledTexture) {
            auto const bound = boundTextures.find(slot.binding);
            if (bound == boundTextures.end()) {
                throw DeviceError{"execute: texture '" + name + " is not bound"};
            }
            auto const &texture = bound->second;
            imageInfos.push_back(VkDescriptorImageInfo{samplerFor(texture.filter, texture.wrap),
                                                       images.at(texture.key).view.get(), VK_IMAGE_LAYOUT_GENERAL});
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &imageInfos.back();
        } else if (slot.kind == ResourceKind::ExternalTexture) {
            auto const &frame = externals.at(boundExternals.at(slot.binding));
            imageInfos.push_back(VkDescriptorImageInfo{frame.sampler.get(), frame.view.get(),
                                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &imageInfos.back();
        } else {
            auto const bound = boundImages.find(slot.binding);
            if (bound == boundImages.end()) {
                throw DeviceError{"execute: image '" + name + " is not bound"};
            }
            imageInfos.push_back(
                VkDescriptorImageInfo{VK_NULL_HANDLE, images.at(bound->second).view.get(), VK_IMAGE_LAYOUT_GENERAL});
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write.pImageInfo = &imageInfos.back();
        }
        writes.push_back(write);
    }
    vkUpdateDescriptorSets(device.handle, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

    auto const layout = pipeline.layout.get();
    auto const cmd = recording();
    for (auto const image : frames) {
        acquireForeignFrame(cmd, image);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline.get());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    if (reflection.pushConstantSize > 0) {
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, reflection.pushConstantSize,
                           pushConstants.data());
    }
    vkCmdDispatch(cmd, dispatch.workGroups.x, dispatch.workGroups.y, dispatch.workGroups.z);
}

} // namespace

std::unique_ptr<Device> createDevice(VulkanDeviceConfig const &config) {
    return std::make_unique<VulkanDevice>(config);
}

} // namespace tc::vulkan
