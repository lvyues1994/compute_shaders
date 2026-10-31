#include <tc/vulkan/vulkan_device.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <atomic>
#include <cstring>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <tc/detail/aliasing.hpp>
#include <tc/detail/hazards.hpp>

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
        {"r16f", VK_FORMAT_R16_SFLOAT},
        {"rg16f", VK_FORMAT_R16G16_SFLOAT},
        {"rgba16f", VK_FORMAT_R16G16B16A16_SFLOAT},
    };
    auto const qualifier = glslFormatQualifier(format);
    for (auto const &entry : table) {
        if (qualifier == entry.qualifier) {
            return entry.format;
        }
    }
    throw DeviceError{"image format " + qualifier + " is not supported by the Vulkan backend"};
}

VkFormat vkCompressedFormat(CompressedFormat const format) {
    switch (format) {
    case CompressedFormat::Bc1: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case CompressedFormat::Bc3: return VK_FORMAT_BC3_UNORM_BLOCK;
    case CompressedFormat::Bc4: return VK_FORMAT_BC4_UNORM_BLOCK;
    case CompressedFormat::Bc5: return VK_FORMAT_BC5_UNORM_BLOCK;
    case CompressedFormat::Bc7: return VK_FORMAT_BC7_UNORM_BLOCK;
    case CompressedFormat::Etc2Rgb8: return VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
    case CompressedFormat::Etc2Rgb8A1: return VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK;
    case CompressedFormat::Etc2Rgba8: return VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK;
    case CompressedFormat::EacR11: return VK_FORMAT_EAC_R11_UNORM_BLOCK;
    case CompressedFormat::EacRg11: return VK_FORMAT_EAC_R11G11_UNORM_BLOCK;
    case CompressedFormat::Astc4x4: return VK_FORMAT_ASTC_4x4_UNORM_BLOCK;
    case CompressedFormat::Astc5x5: return VK_FORMAT_ASTC_5x5_UNORM_BLOCK;
    case CompressedFormat::Astc6x6: return VK_FORMAT_ASTC_6x6_UNORM_BLOCK;
    case CompressedFormat::Astc8x8: return VK_FORMAT_ASTC_8x8_UNORM_BLOCK;
    case CompressedFormat::Astc10x10: return VK_FORMAT_ASTC_10x10_UNORM_BLOCK;
    case CompressedFormat::Astc12x12: return VK_FORMAT_ASTC_12x12_UNORM_BLOCK;
    }
    return VK_FORMAT_UNDEFINED;
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

std::uint32_t timestampValidBits(VkPhysicalDevice const physical, std::uint32_t const family) {
    auto count = std::uint32_t{0};
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    auto families = std::vector<VkQueueFamilyProperties>(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    return family < count ? families[family].timestampValidBits : 0;
}

// Every buffer may be a storage buffer, a transfer source or target, and
// hold the arguments of an indirect dispatch.
constexpr VkBufferUsageFlags kBufferUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                          | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

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
    bool float64 = false;
    bool syncFdImport = false; // VK_KHR_external_semaphore_fd with sync files
    bool syncFdExport = false;
    bool int64 = false;
    bool int64Atomics = false;
    bool bc = false; // texture compression families enabled on the device
    bool etc2 = false;
    bool astc = false;
    std::uint32_t subgroupSize = 0; // with every GL_KHR_shader_subgroup operation in compute shaders; else 0
    bool timelineSemaphores = false;
    bool memoryImport = false; // opaque fds of other devices' memory (VK_KHR_external_memory_fd)
    std::optional<GpuIdentity> gpu;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    std::uint32_t queueFamily = 0;
    DeviceOwner device;
    VkQueue queue = VK_NULL_HANDLE;
    // The transfer queue, when the device has a second queue for it.
    std::optional<std::uint32_t> transferFamily;
    std::uint32_t transferIndex = 0;
    VkQueue transferQueue = VK_NULL_HANDLE;
};

// A queue for TransferQueue: of a family that only copies (a DMA engine)
// with whole-texel copy granularity, else the second queue of the compute
// family; nullopt when there is neither.
std::optional<std::pair<std::uint32_t, std::uint32_t>> transferQueueOf(VkPhysicalDevice const physical,
                                                                       std::uint32_t const computeFamily) {
    auto count = std::uint32_t{0};
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, nullptr);
    auto families = std::vector<VkQueueFamilyProperties>(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, families.data());
    for (std::uint32_t i = 0; i < count; ++i) {
        auto const flags = families[i].queueFlags;
        auto const &granularity = families[i].minImageTransferGranularity;
        if ((flags & VK_QUEUE_TRANSFER_BIT) != 0 and (flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == 0
            and families[i].queueCount > 0 and granularity.width == 1 and granularity.height == 1
            and granularity.depth == 1) {
            return std::pair{i, 0u};
        }
    }
    if (families[computeFamily].queueCount >= 2) {
        return std::pair{computeFamily, 1u};
    }
    return std::nullopt;
}

// A buffer in host-visible device memory, persistently mapped.
struct MappedBuffer {
    std::shared_ptr<VulkanContext> context; // declared first: outlives the handles below
    DeviceObject<VkDeviceMemory> memory;
    DeviceObject<VkBuffer> buffer;
    std::byte *mapped = nullptr;
};

// A memory heap; placed buffers and images keep its memory alive.
struct HeapMirror {
    std::weak_ptr<void const> owner;
    std::shared_ptr<DeviceObject<VkDeviceMemory>> memory;
    std::uint32_t memoryType = 0;
    VkDeviceSize size = 0;
};

struct BufferMirror {
    std::weak_ptr<void const> owner;
    std::shared_ptr<DeviceObject<VkDeviceMemory>> heap; // placed buffers: instead of `memory`
    DeviceObject<VkDeviceMemory> memory; // declared before the buffer: freed after it
    DeviceObject<VkBuffer> buffer;
    std::shared_ptr<MappedBuffer> mapped; // set for createBuffer() allocations instead of the two above
    VkDeviceSize size = 0;
    bool imported = false; // memory of another device: taken from and given back to VK_QUEUE_FAMILY_EXTERNAL

    VkBuffer handle() const noexcept { return mapped ? mapped->buffer.get() : buffer.get(); }
};

struct ImageMirror {
    std::weak_ptr<void const> owner;
    std::shared_ptr<DeviceObject<VkDeviceMemory>> heap; // placed images: instead of `memory`
    DeviceObject<VkDeviceMemory> memory;
    DeviceObject<VkImage> image;
    DeviceObject<VkImageView> view; // every level: sampled textures
    std::map<std::uint32_t, DeviceObject<VkImageView>> levelViews; // one level: storage images
    ImageFormat format;
    Dim dim = Dim::D2;
    ivec3 extent{};
    std::uint32_t levels = 1;
    VkImageUsageFlags usage = 0;
    std::optional<CompressedFormat> compressed; // a block-compressed texture: `format` is unused
};

std::uint32_t layerCount(Dim const dim, ivec3 const extent) {
    return dim == Dim::D2Array ? static_cast<std::uint32_t>(extent.z) : 1u;
}

// Texel extent of one level as Vulkan counts it: array layers are not depth.
VkExtent3D levelExtent(Dim const dim, ivec3 const extent, std::uint32_t const level) {
    auto const size = mipExtent(dim, extent, level);
    return VkExtent3D{static_cast<std::uint32_t>(size.x), static_cast<std::uint32_t>(size.y),
                      dim == Dim::D3 ? static_cast<std::uint32_t>(size.z) : 1u};
}

std::size_t levelBytes(ImageTransfer const &transfer, std::uint32_t const level) {
    auto const size = mipExtent(transfer.dim, transfer.extent, level);
    return static_cast<std::size_t>(size.x) * size.y * size.z * transfer.format.bytesPerTexel();
}

VkBufferImageCopy imageCopy(ImageTransfer const &transfer, std::uint32_t const level, VkDeviceSize const offset) {
    auto region = VkBufferImageCopy{};
    region.bufferOffset = offset;
    region.imageSubresource =
        VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, level, 0, layerCount(transfer.dim, transfer.extent)};
    region.imageExtent = levelExtent(transfer.dim, transfer.extent, level);
    return region;
}

struct BoundTexture {
    void const *key = nullptr;
    Filter filter = Filter::Nearest;
    Wrap wrap = Wrap::ClampToEdge;
    Mip mip = Mip::None;
};

struct BoundImage {
    void const *key = nullptr;
    std::uint32_t level = 0;
};

struct BoundBuffer {
    void const *key = nullptr;
    std::optional<ByteRange> range;
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

// A platform frame imported as a storage image.
struct ExternalImageMirror {
    std::weak_ptr<void const> owner;
    DeviceObject<VkDeviceMemory> memory;
    DeviceObject<VkImage> image;
    DeviceObject<VkImageView> view;
    bool acquired = false; // after the first use: in GENERAL layout
};

// Passes a storage frame between the foreign queue family (compositor,
// camera, CPU) and ours in GENERAL layout, which keeps its contents. A
// newly imported image starts from UNDEFINED.
void transferForeignImage(VkCommandBuffer const cmd, VkImage const image, std::uint32_t const queueFamily,
                          bool const acquire, VkImageLayout const oldLayout = VK_IMAGE_LAYOUT_GENERAL) {
    auto barrier = vkStruct<VkImageMemoryBarrier>(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER);
    barrier.srcAccessMask = acquire ? 0 : VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = acquire ? VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT : 0;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = acquire ? VK_QUEUE_FAMILY_FOREIGN_EXT : queueFamily;
    barrier.dstQueueFamilyIndex = acquire ? queueFamily : VK_QUEUE_FAMILY_FOREIGN_EXT;
    barrier.image = image;
    barrier.subresourceRange = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, acquire ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         acquire ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &barrier);
}

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
    case ResourceKind::ExternalImage:
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

// Host-visible memory for copies to and from device-local resources.
struct StagingChunk {
    DeviceObject<VkDeviceMemory> memory; // declared before the buffer: freed after it
    DeviceObject<VkBuffer> buffer;
    std::byte *mapped = nullptr;
    VkDeviceSize capacity = 0;
    VkDeviceSize used = 0;
};

// Staging bytes a command reads or writes.
struct StagingSlice {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    std::byte *mapped = nullptr; // the host address of `offset`
};

// The fence of one submitted batch. It keeps the Vulkan device alive, so a
// Submission may outlive the VulkanDevice. Fences are never reset, so any
// thread may query or wait.
struct FenceCompletion final : Submission::Completion {
    FenceCompletion(std::shared_ptr<VulkanContext> context_, DeviceObject<VkFence> fence_, VkSemaphore const timeline_,
                    std::uint64_t const value_) noexcept
        : context(std::move(context_)), fence(std::move(fence_)), timeline(timeline_), value(value_) {}

    bool isComplete() const override {
        if (signaled.load(std::memory_order_acquire)) {
            return true;
        }
        auto const status = vkGetFenceStatus(context->device.handle, fence.get());
        if (status == VK_NOT_READY) {
            return false;
        }
        check(status, "vkGetFenceStatus");
        signaled.store(true, std::memory_order_release);
        return true;
    }

    void wait() const override {
        if (signaled.load(std::memory_order_acquire)) {
            return;
        }
        auto const handle = fence.get();
        check(vkWaitForFences(context->device.handle, 1, &handle, VK_TRUE, UINT64_MAX), "vkWaitForFences");
        signaled.store(true, std::memory_order_release);
    }

    std::shared_ptr<VulkanContext> context; // declared first: destroyed after the fence
    DeviceObject<VkFence> fence;
    // The timeline semaphore the submission signals with `value`, so the
    // other queue can wait for it; VK_NULL_HANDLE without a transfer queue.
    VkSemaphore timeline;
    std::uint64_t value;
    mutable std::atomic<bool> signaled{false};
};

struct PendingTiming {
    std::string program;
    uvec3 workGroups{};
    std::uint32_t firstQuery = 0; // start and end timestamps
    bool indirect = false;
};

// Everything one command buffer of work uses on the device: reused once
// the device has finished it, so the host records the next batch while the
// device runs earlier ones.
struct Batch {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE; // from the device's command pool
    std::vector<DeviceObject<VkDescriptorPool>> descriptorPools;
    std::size_t descriptorPool = 0; // the pool new sets come from
    std::vector<StagingChunk> staging;
    DeviceObject<VkQueryPool> queryPool;
    std::uint32_t queriesUsed = 0;
    std::vector<PendingTiming> pendingTimings;
    std::vector<DeviceObject<VkSemaphore>> semaphores; // waited on or signalled by the submission
    std::vector<VkBuffer> imported;              // acquired from VK_QUEUE_FAMILY_EXTERNAL, released at the end
    std::uint64_t serial = 0;                    // submission order
    std::uint64_t submission = 0;                // the Submission::sequence() it is part of
    std::shared_ptr<FenceCompletion> completion; // set while in flight
};

// A device object released while queued work may still use it.
struct Grave {
    std::uint64_t serial = 0;         // destroyed once this batch has completed
    std::uint64_t transferSerial = 0; // and this batch of the transfer queue
    std::shared_ptr<void> object;
};

struct VulkanDevice final : Device {
    explicit VulkanDevice(VulkanDeviceConfig const &config);
    VulkanDevice(VulkanDevice const &) = delete;
    VulkanDevice &operator=(VulkanDevice const &) = delete;
    ~VulkanDevice() override;

    DeviceInfo info() const override {
        auto const memory = sharedMemoryType();
        auto result = DeviceInfo{};
        result.backend = "vulkan";
        result.deviceName = properties.deviceName;
        result.unifiedMemory =
            (memoryProperties.memoryTypes[memory].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        result.timers = timestampBits > 0 and properties.limits.timestampPeriod > 0.0f;
        result.syncFiles = context->syncFdImport and context->syncFdExport;
        result.placedImages = true;
        result.separateTransfers = transferQueue != nullptr;
        result.gpu = context->gpu;
        result.importsMemory = context->memoryImport;
        auto const &supported = properties.limits;
        auto &limits = result.limits;
        limits.storageBuffers = supported.maxPerStageDescriptorStorageBuffers;
        limits.storageImages = supported.maxPerStageDescriptorStorageImages;
        limits.sampledTextures = std::min(supported.maxPerStageDescriptorSampledImages, supported.maxPerStageDescriptorSamplers);
        limits.storageBufferBytes = supported.maxStorageBufferRange;
        limits.sharedBytes = supported.maxComputeSharedMemorySize;
        limits.workGroupInvocations = supported.maxComputeWorkGroupInvocations;
        limits.workGroupSize =
            uvec3{supported.maxComputeWorkGroupSize[0], supported.maxComputeWorkGroupSize[1], supported.maxComputeWorkGroupSize[2]};
        limits.workGroupCount = uvec3{supported.maxComputeWorkGroupCount[0], supported.maxComputeWorkGroupCount[1],
                                      supported.maxComputeWorkGroupCount[2]};
        limits.pushConstantBytes = supported.maxPushConstantsSize;
        limits.storageBufferOffsetAlignment = static_cast<std::uint32_t>(supported.minStorageBufferOffsetAlignment);
        limits.features = (context->float64 ? static_cast<std::uint32_t>(KernelFeature::Double) : 0u)
                        | (context->subgroupSize != 0 ? static_cast<std::uint32_t>(KernelFeature::Subgroups) : 0u)
                        | (context->int64 ? static_cast<std::uint32_t>(KernelFeature::Int64) : 0u)
                        | (context->int64Atomics ? static_cast<std::uint32_t>(KernelFeature::Int64Atomics) : 0u);
        limits.subgroupSize = context->subgroupSize;
        limits.imageDimension2D = supported.maxImageDimension2D;
        limits.imageDimension3D = supported.maxImageDimension3D;
        limits.imageArrayLayers = supported.maxImageArrayLayers;
        for (std::uint32_t i = 0; i < kCompressedFormatCount; ++i) {
            auto const format = static_cast<CompressedFormat>(i);
            auto const family = i <= static_cast<std::uint32_t>(CompressedFormat::Bc7)      ? context->bc
                              : i <= static_cast<std::uint32_t>(CompressedFormat::EacRg11) ? context->etc2
                                                                                          : context->astc;
            if (family and (formatFeatures(vkCompressedFormat(format)) & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0) {
                limits.compressedFormats |= compressedBit(format);
                auto image = VkImageFormatProperties{};
                if (vkGetPhysicalDeviceImageFormatProperties(physical, vkCompressedFormat(format), VK_IMAGE_TYPE_2D,
                                                             VK_IMAGE_TILING_OPTIMAL,
                                                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                                             0, &image)
                        == VK_SUCCESS
                    and image.maxArrayLayers > 1) {
                    limits.compressedArrayFormats |= compressedBit(format);
                }
            }
        }
        return result;
    }

protected:
    std::optional<ExternalMemory> doAllocateBuffer(ResourceIdentity const &identity, std::size_t bytes) override;
    void doUploadBuffer(ResourceIdentity const &identity, ConstByteSpan bytes, ByteRange range) override;
    void doDownloadBuffer(ResourceIdentity const &identity, ByteSpan bytes, ByteRange range) override;
    std::optional<std::string> doCheckKernel(KernelProgram const &program) const override {
        if (program.spirv.empty()) {
            return "kernel '" + std::string{program.name} + "' has no SPIR-V";
        }
        return std::nullopt;
    }
    void doUploadImage(ImageTransfer const &transfer) override;
    void doUploadCompressed(CompressedTransfer const &transfer) override;
    void doDownloadImage(ImageTransfer const &transfer) override;
    bool doCreateImage(ImageCreation const &request) override;
    void doCopyBufferToImage(BufferImageCopy const &copy) override;
    bool doCreateHeap(HeapCreation const &request) override;
    bool doCreateDeviceBuffer(BufferCreation const &request) override;
    PlacementSize doBufferPlacement(std::size_t bytes) override;
    PlacementSize doImagePlacement(ImageCreation const &request) override;
    void doCopyBuffer(BufferCopy const &copy) override;
    void doImportBuffer(BufferImport const &request) override;
    void doUseKernel(KernelUse const &use) override;
    void doBindBuffer(BufferBindRequest const &request) override;
    void doBindImage(ImageBindRequest const &request) override;
    void doBindTexture(TextureBindRequest const &request) override;
    void doBindExternalTexture(ExternalTextureBindRequest const &request) override;
#if defined(__ANDROID__)
    void doBindExternalImage(ExternalImageBindRequest const &request) override;
#endif
    void doWaitFor(SyncFence fence) override;
    void doWaitForSubmission(Submission const &submission) override;
    TransferQueue &doTransfers() override;
    FencedSubmission doSubmitWithFence() override;
    void doSetUniform(UniformWrite const &write) override;
    void doExecute(Dispatch const &dispatch) override;
    Submission doSubmit() override { return submitBatch(); }
    void doFinish() override { flush(); }
    std::vector<KernelTiming> doTakeTimings(std::uint64_t last) override;

private:
    static constexpr std::uint32_t kQueryCapacity = 256;

    struct Transfers; // the TransferQueue on the device's second queue

    VkFormatFeatureFlags formatFeatures(VkFormat format) const;
    // Buffers and images both queues use are shared between their queue
    // families, so neither has to hand them over.
    template <class Info>
    void shareAcrossQueues(Info &info) const {
        if (concurrentFamilies[0] != concurrentFamilies[1]) {
            info.sharingMode = VK_SHARING_MODE_CONCURRENT;
            info.queueFamilyIndexCount = 2;
            info.pQueueFamilyIndices = concurrentFamilies.data();
        }
    }
    DeviceObject<VkSemaphore> makeTimelineSemaphore();
    // The region of a copyBufferToImage, from byte `offset` of its buffer.
    static VkBufferImageCopy regionOf(BufferImageCopy const &copy);
    VkSampler samplerFor(Filter filter, Wrap wrap, Mip mip);
    DeviceObject<VkImageView> createView(ImageMirror const &mirror, std::uint32_t first, std::uint32_t count) const;
    // The image tc creates for this format and shape: every usage the format
    // supports, since one image may be stored, sampled and copied.
    VkImageCreateInfo imageInfo(ImageFormat const &format, std::optional<CompressedFormat> compressed, Dim dim,
                                ivec3 extent, std::uint32_t levels) const;
    // Replaces `mirror` with a new image in GENERAL layout (contents
    // undefined); a placed image takes its layout at its first use.
    void allocateImage(ImageMirror &mirror, ResourceIdentity const &identity, ImageFormat const &format,
                       std::optional<CompressedFormat> compressed, Dim dim, ivec3 extent, std::uint32_t levels,
                       std::optional<HeapRange> const &placement = std::nullopt);
    // Memory of `heap` for a resource with `requirements` at `offset`.
    std::shared_ptr<DeviceObject<VkDeviceMemory>> heapMemory(HeapRange const &placement,
                                                             VkMemoryRequirements const &requirements,
                                                             char const *what) const;
    PlacementSize placementOf(VkMemoryRequirements const &requirements) const;
    detail::ResourceAccess access(void const *const key, std::size_t const offset, std::size_t const size,
                                  bool const read, bool const write) const {
        return aliases.access(key, offset, size, read, write);
    }
    // Before `cmd` uses a placed resource whose bytes an alias used last:
    // an image's contents are discarded into GENERAL layout. An imported
    // buffer is acquired from the other device for the rest of the batch.
    void claim(VkCommandBuffer cmd, void const *key);
    VkImageView levelView(ImageMirror &mirror, std::uint32_t level);
    void createInstance(VulkanDeviceConfig const &config);
    void pickPhysicalDevice(VulkanDeviceConfig const &config);
    void createLogicalDevice();
    std::uint32_t memoryType(std::uint32_t typeBits, VkMemoryPropertyFlags required) const;
    std::uint32_t sharedMemoryType() const;
    DeviceObject<VkDeviceMemory> allocate(VkMemoryRequirements const &requirements, VkMemoryPropertyFlags flags);
    // Commands go into the recording batch, begun on first use, until
    // submitBatch() sends it; flush() also waits for every batch.
    Batch &recordingBatch();
    VkCommandBuffer recording() { return recordingBatch().commandBuffer; }
    // The command buffer for a command with `accesses`, after a barrier when
    // an earlier command conflicts with them.
    VkCommandBuffer commandFor(std::vector<detail::ResourceAccess> const &accesses);
    Submission submitBatch(DeviceObject<VkSemaphore> signal = {});
    DeviceObject<VkSemaphore> makeSemaphore(bool exportable);
    void flush();
    void retireCompleted();
    template <class T>
    void bury(T &&object);
    StagingSlice stagingSlice(VkDeviceSize size);
    StagingChunk makeStaging(VkDeviceSize capacity);
    DeviceObject<VkDescriptorPool> makeDescriptorPool();
    // Query indices of a start and an end timestamp in the recording batch.
    std::uint32_t reserveTimestamps(VkCommandBuffer cmd);
    void readTimestamps(Batch &batch);
    VkDescriptorSet allocateSet(VkDescriptorSetLayout layout);
    void collectGarbage();
    KernelProgram const &programFor(std::string_view name);
    PipelineObject &pipelineFor(KernelProgram const &program, std::vector<VkSampler> const &immutableSamplers);
    void acquireForeignFrame(VkCommandBuffer cmd, VkImage image) const {
        vulkan::acquireForeignFrame(cmd, image, queueFamily);
    }
#if defined(__ANDROID__)
    void importFrame(ExternalMirror &mirror, ExternalImage const &image);
    void importStorageFrame(ExternalImageMirror &mirror, ExternalImage const &image, ImageFormat const &format);
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
    DeviceObject<VkCommandPool> commandPool; // declared before the batches: frees their command buffers after them
    std::unique_ptr<Batch> recordingNow;     // recorded, not yet submitted
    std::deque<std::unique_ptr<Batch>> inFlight; // submitted, oldest first
    std::vector<std::unique_ptr<Batch>> idleBatches;
    std::uint64_t nextSerial = 1;
    std::uint64_t completedSerial = 0; // every batch up to this one has completed
    std::vector<Grave> graveyard;
    detail::HazardTracker hazards;
    detail::AliasTracker aliases;
    std::vector<DeviceObject<VkSemaphore>> pendingWaits; // for the next submission
    PFN_vkGetSemaphoreFdKHR getSemaphoreFd = nullptr;
    PFN_vkImportSemaphoreFdKHR importSemaphoreFd = nullptr;
    std::unordered_map<void const *, HeapMirror> heaps;
    std::unordered_map<void const *, BufferMirror> buffers;
    std::unordered_map<void const *, ImageMirror> images;
    std::unordered_map<void const *, ExternalMirror> externals;
    std::unordered_map<void const *, ExternalImageMirror> externalImages;
    std::unordered_map<std::string, KernelProgram> programs;
    std::unordered_map<std::string, PipelineObject> pipelines;
    std::map<std::tuple<Filter, Wrap, Mip>, DeviceObject<VkSampler>> samplers;

    KernelProgram const *current = nullptr;
    std::map<std::uint32_t, BoundBuffer> boundBuffers;
    std::map<std::uint32_t, BoundImage> boundImages;
    std::map<std::uint32_t, BoundTexture> boundTextures;
    std::map<std::uint32_t, void const *> boundExternals;
    std::map<std::uint32_t, void const *> boundExternalImages;
    std::map<std::uint32_t, bool> uniformsSet;
    std::array<std::byte, 256> pushConstants{};

    std::uint32_t timestampBits = 0; // of the compute queue family; 0: no timestamps
    std::vector<KernelTiming> timings;

    // With a transfer queue: each batch signals its serial on `mainTimeline`,
    // and batches wait for transfer batches through theirs.
    std::array<std::uint32_t, 2> concurrentFamilies{};
    DeviceObject<VkSemaphore> mainTimeline;
    std::vector<std::pair<VkSemaphore, std::uint64_t>> pendingTimelineWaits; // for the next submission
    std::unique_ptr<Transfers> transferQueue;
};

// Copies on the device's second queue, in batches like the device's own,
// each signalling its serial on `timeline`.
struct VulkanDevice::Transfers final : TransferQueue {
    explicit Transfers(VulkanDevice &device_) : TransferQueue(device_), device(device_) {
        auto const handle = device.device.handle;
        auto poolInfo = vkStruct<VkCommandPoolCreateInfo>(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = *device.context->transferFamily;
        auto rawPool = VkCommandPool{};
        check(vkCreateCommandPool(handle, &poolInfo, nullptr, &rawPool), "vkCreateCommandPool (transfers)");
        pool = DeviceObject<VkCommandPool>{handle, rawPool, vkDestroyCommandPool};
        timeline = device.makeTimelineSemaphore();
    }

    // The last batch that may use what is recorded or sent now.
    std::uint64_t latestSerial() const noexcept { return recording != nullptr ? recording->serial : nextSerial - 1; }
    std::uint64_t completed() const noexcept { return completedSerial; }

    // Recycles finished batches; returns the serial every batch up to which has completed.
    std::uint64_t retire() {
        while (not inFlight.empty() and inFlight.front()->completion->isComplete()) {
            completedSerial = inFlight.front()->serial;
            inFlight.front()->completion.reset();
            idle.push_back(std::move(inFlight.front()));
            inFlight.pop_front();
        }
        return completedSerial;
    }

    void flush() {
        doSubmit();
        for (auto const &batch : inFlight) {
            batch->completion->wait();
        }
        retire();
    }

protected:
    void doCopyBuffer(BufferCopy const &copy) override {
        auto const source = device.buffers.find(copy.source->key());
        auto const target = device.buffers.find(copy.target->key());
        if (source == device.buffers.end() or target == device.buffers.end()) {
            throw DeviceError{"copyBuffer: a buffer was never uploaded to this device"};
        }
        if (source->second.imported or target->second.imported) {
            throw DeviceError{"copyBuffer: imported buffers belong to the device's own timeline, not transfers()"};
        }
        auto const cmd = commandFor({device.access(copy.source->key(), copy.sourceOffset, copy.bytes, true, false),
                                     device.access(copy.target->key(), copy.targetOffset, copy.bytes, false, true)});
        device.claim(cmd, copy.source->key());
        device.claim(cmd, copy.target->key());
        auto const region = VkBufferCopy{copy.sourceOffset, copy.targetOffset, copy.bytes};
        vkCmdCopyBuffer(cmd, source->second.handle(), target->second.handle(), 1, &region);
    }

    void doCopyBufferToImage(BufferImageCopy const &copy) override {
        auto const buffer = device.buffers.find(copy.buffer->key());
        if (buffer == device.buffers.end() or buffer->second.size < copy.offset + copy.bytes) {
            throw DeviceError{"copyBufferToImage: the buffer was never uploaded to this device (or changed size)"};
        }
        if (buffer->second.imported) {
            throw DeviceError{"copyBufferToImage: imported buffers belong to the device's own timeline, not transfers()"};
        }
        auto const &mirror = device.images.at(copy.image->key());
        auto const region = regionOf(copy);
        auto const cmd = commandFor({device.access(copy.buffer->key(), copy.offset, copy.bytes, true, false),
                                     device.access(copy.image->key(), 0, 0, false, true)});
        device.claim(cmd, copy.buffer->key());
        device.claim(cmd, copy.image->key());
        vkCmdCopyBufferToImage(cmd, buffer->second.handle(), mirror.image.get(), VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    }

    void doWaitFor(Submission const &submission) override {
        auto const *completion = dynamic_cast<FenceCompletion const *>(completionOf(submission));
        if (completion != nullptr and completion->timeline == device.mainTimeline.get()) {
            doSubmit(); // copies queued before this call do not wait
            pendingWaits.emplace_back(completion->timeline, completion->value);
        }
    }

    Submission doSubmit() override {
        if (recording == nullptr) {
            if (pendingWaits.empty()) {
                return inFlight.empty() ? Submission{} : Submission{inFlight.back()->completion};
            }
            recordingBatch(); // an empty batch carries the waits
        }
        auto &batch = *recording;
        hostVisibleBarrier(batch.commandBuffer);
        check(vkEndCommandBuffer(batch.commandBuffer), "vkEndCommandBuffer (transfers)");
        auto const handle = device.device.handle;
        auto fenceInfo = vkStruct<VkFenceCreateInfo>(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
        auto rawFence = VkFence{};
        check(vkCreateFence(handle, &fenceInfo, nullptr, &rawFence), "vkCreateFence");
        auto fence = DeviceObject<VkFence>{handle, rawFence, vkDestroyFence};
        auto waits = std::vector<VkSemaphore>{};
        auto waitValues = std::vector<std::uint64_t>{};
        for (auto const &[semaphore, value] : pendingWaits) {
            waits.push_back(semaphore);
            waitValues.push_back(value);
        }
        auto const stages = std::vector<VkPipelineStageFlags>(waits.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        auto const signal = timeline.get();
        auto timelines = vkStruct<VkTimelineSemaphoreSubmitInfo>(VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO);
        timelines.waitSemaphoreValueCount = static_cast<std::uint32_t>(waitValues.size());
        timelines.pWaitSemaphoreValues = waitValues.data();
        timelines.signalSemaphoreValueCount = 1;
        timelines.pSignalSemaphoreValues = &batch.serial;
        auto info = vkStruct<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
        info.pNext = &timelines;
        info.waitSemaphoreCount = static_cast<std::uint32_t>(waits.size());
        info.pWaitSemaphores = waits.data();
        info.pWaitDstStageMask = stages.data();
        info.commandBufferCount = 1;
        info.pCommandBuffers = &batch.commandBuffer;
        info.signalSemaphoreCount = 1;
        info.pSignalSemaphores = &signal;
        check(vkQueueSubmit(device.context->transferQueue, 1, &info, rawFence), "vkQueueSubmit (transfers)");
        pendingWaits.clear();
        batch.completion = std::make_shared<FenceCompletion>(device.context, std::move(fence), signal, batch.serial);
        auto submission = Submission{batch.completion};
        inFlight.push_back(std::move(recording));
        return submission;
    }

private:
    struct Batch {
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE; // from `pool`
        std::uint64_t serial = 0;
        std::shared_ptr<FenceCompletion> completion; // set while in flight
    };

    Batch &recordingBatch() {
        if (recording != nullptr) {
            return *recording;
        }
        retire();
        if (idle.empty()) {
            auto batch = std::make_unique<Batch>();
            auto info = vkStruct<VkCommandBufferAllocateInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
            info.commandPool = pool.get();
            info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            info.commandBufferCount = 1;
            check(vkAllocateCommandBuffers(device.device.handle, &info, &batch->commandBuffer), "vkAllocateCommandBuffers");
            idle.push_back(std::move(batch));
        }
        recording = std::move(idle.back());
        idle.pop_back();
        recording->serial = nextSerial++;
        check(vkResetCommandBuffer(recording->commandBuffer, 0), "vkResetCommandBuffer");
        auto begin = vkStruct<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(recording->commandBuffer, &begin), "vkBeginCommandBuffer");
        return *recording;
    }

    VkCommandBuffer commandFor(std::vector<detail::ResourceAccess> const &accesses) {
        auto const cmd = recordingBatch().commandBuffer;
        if (hazards.needsBarrier(accesses)) {
            fullBarrier(cmd);
        }
        return cmd;
    }

    VulkanDevice &device;
    DeviceObject<VkCommandPool> pool; // declared before the batches: frees their command buffers after them
    DeviceObject<VkSemaphore> timeline;
    std::unique_ptr<Batch> recording;
    std::deque<std::unique_ptr<Batch>> inFlight;
    std::vector<std::unique_ptr<Batch>> idle;
    std::uint64_t nextSerial = 1;
    std::uint64_t completedSerial = 0;
    detail::HazardTracker hazards;
    std::vector<std::pair<VkSemaphore, std::uint64_t>> pendingWaits; // for the next batch
};

VulkanDevice::~VulkanDevice() {
    try {
        flush();
    } catch (DeviceError const &error) {
        std::fprintf(stderr, "[vulkan] queued work failed at shutdown: %s\n", error.what());
    }
}

TransferQueue &VulkanDevice::doTransfers() {
    return transferQueue != nullptr ? *transferQueue : Device::doTransfers();
}

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
    concurrentFamilies = {queueFamily, context->transferFamily.value_or(queueFamily)};
    if (context->transferQueue != VK_NULL_HANDLE) {
        mainTimeline = makeTimelineSemaphore();
        transferQueue = std::make_unique<Transfers>(*this);
    }
}

DeviceObject<VkSemaphore> VulkanDevice::makeTimelineSemaphore() {
    auto type = vkStruct<VkSemaphoreTypeCreateInfo>(VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO);
    type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    auto info = vkStruct<VkSemaphoreCreateInfo>(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
    info.pNext = &type;
    auto semaphore = VkSemaphore{};
    check(vkCreateSemaphore(device.handle, &info, nullptr, &semaphore), "vkCreateSemaphore (timeline)");
    return DeviceObject<VkSemaphore>{device.handle, semaphore, vkDestroySemaphore};
}

void VulkanDevice::createInstance(VulkanDeviceConfig const &config) {
    auto app = vkStruct<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
    app.pApplicationName = "tc";
    app.pEngineName = "tc";
    // Vulkan 1.1 when the loader has it: sampler YCbCr conversion (external
    // frames) is core there; 1.2 adds timeline semaphores (the transfer
    // queue). Everything else needs only 1.0.
    auto const enumerateVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
        vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
    auto loaderVersion = std::uint32_t{VK_API_VERSION_1_0};
    if (enumerateVersion != nullptr) {
        enumerateVersion(&loaderVersion);
    }
    context->apiVersion = loaderVersion >= VK_API_VERSION_1_2   ? VK_API_VERSION_1_2
                        : loaderVersion >= VK_API_VERSION_1_1 ? VK_API_VERSION_1_1
                                                              : VK_API_VERSION_1_0;
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
    timestampBits = timestampValidBits(physical, queueFamily);
}

void VulkanDevice::createLogicalDevice() {
    auto supported = VkPhysicalDeviceFeatures{};
    vkGetPhysicalDeviceFeatures(physical, &supported);
    auto features = VkPhysicalDeviceFeatures{};
    features.shaderFloat64 = supported.shaderFloat64;
    context->float64 = supported.shaderFloat64 == VK_TRUE;
    features.shaderInt64 = supported.shaderInt64;
    context->int64 = supported.shaderInt64 == VK_TRUE;
    features.textureCompressionBC = supported.textureCompressionBC;
    features.textureCompressionETC2 = supported.textureCompressionETC2;
    features.textureCompressionASTC_LDR = supported.textureCompressionASTC_LDR;
    context->bc = supported.textureCompressionBC == VK_TRUE;
    context->etc2 = supported.textureCompressionETC2 == VK_TRUE;
    context->astc = supported.textureCompressionASTC_LDR == VK_TRUE;
    features.shaderStorageImageExtendedFormats = supported.shaderStorageImageExtendedFormats;

    auto extensions = std::vector<char const *>{};
    if (hasDeviceExtension(physical, "VK_KHR_portability_subset")) {
        extensions.push_back("VK_KHR_portability_subset");
    }
    auto info = vkStruct<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
    auto const vulkan11 = context->apiVersion >= VK_API_VERSION_1_1 and properties.apiVersion >= VK_API_VERSION_1_1;
    // Subgroup operations need Vulkan 1.1 (SPIR-V 1.3) and every operation
    // class tc exposes, in compute shaders.
    if (vulkan11) {
        auto ids = vkStruct<VkPhysicalDeviceIDProperties>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES);
        auto subgroups = vkStruct<VkPhysicalDeviceSubgroupProperties>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES);
        subgroups.pNext = &ids;
        auto properties2 = vkStruct<VkPhysicalDeviceProperties2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
        properties2.pNext = &subgroups;
        vkGetPhysicalDeviceProperties2(physical, &properties2);
        auto identity = GpuIdentity{};
        std::copy(std::begin(ids.deviceUUID), std::end(ids.deviceUUID), identity.device.begin());
        std::copy(std::begin(ids.driverUUID), std::end(ids.driverUUID), identity.driver.begin());
        context->gpu = identity;
        // Memory of other devices of this GPU, as opaque file descriptors.
        if (hasDeviceExtension(physical, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME)) {
            extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
            context->memoryImport = true;
        }
        auto constexpr needed = VkSubgroupFeatureFlags{
            VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT
            | VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT};
        if ((subgroups.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0
            and (subgroups.supportedOperations & needed) == needed) {
            context->subgroupSize = subgroups.subgroupSize;
        }
    }
    // 64-bit atomics on buffer elements and timeline semaphores: Vulkan 1.2
    // features. With a 1.1 API, 64-bit atomics are an extension.
    auto const vulkan12 = context->apiVersion >= VK_API_VERSION_1_2 and properties.apiVersion >= VK_API_VERSION_1_2;
    auto vulkan12Features = vkStruct<VkPhysicalDeviceVulkan12Features>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
    auto atomicInt64 =
        vkStruct<VkPhysicalDeviceShaderAtomicInt64Features>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES);
    void *chain = nullptr;
    if (vulkan12) {
        auto supported12 = vkStruct<VkPhysicalDeviceVulkan12Features>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
        auto features2 = vkStruct<VkPhysicalDeviceFeatures2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
        features2.pNext = &supported12;
        vkGetPhysicalDeviceFeatures2(physical, &features2);
        if (context->int64 and supported12.shaderBufferInt64Atomics == VK_TRUE) {
            vulkan12Features.shaderBufferInt64Atomics = VK_TRUE;
            context->int64Atomics = true;
        }
        if (supported12.timelineSemaphore == VK_TRUE) {
            vulkan12Features.timelineSemaphore = VK_TRUE;
            context->timelineSemaphores = true;
        }
        chain = &vulkan12Features;
    } else if (vulkan11 and context->int64 and hasDeviceExtension(physical, VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME)) {
        auto features2 = vkStruct<VkPhysicalDeviceFeatures2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
        features2.pNext = &atomicInt64;
        vkGetPhysicalDeviceFeatures2(physical, &features2);
        if (atomicInt64.shaderBufferInt64Atomics == VK_TRUE) {
            atomicInt64.shaderSharedInt64Atomics = VK_FALSE;
            atomicInt64.pNext = nullptr;
            chain = &atomicInt64;
            extensions.push_back(VK_KHR_SHADER_ATOMIC_INT64_EXTENSION_NAME);
            context->int64Atomics = true;
        }
    }
    info.pNext = chain;
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
            ycbcr.pNext = chain;
            info.pNext = &ycbcr;
            context->externalImages = true;
        }
    }
#endif
    // Sync files: waits on fences of other APIs and processes, and fences
    // for them (Vulkan 1.1 external semaphores).
    if (context->apiVersion >= VK_API_VERSION_1_1 and properties.apiVersion >= VK_API_VERSION_1_1
        and hasDeviceExtension(physical, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME)) {
        auto semaphoreInfo =
            vkStruct<VkPhysicalDeviceExternalSemaphoreInfo>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO);
        semaphoreInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
        auto semaphoreProperties = vkStruct<VkExternalSemaphoreProperties>(VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES);
        vkGetPhysicalDeviceExternalSemaphoreProperties(physical, &semaphoreInfo, &semaphoreProperties);
        auto const features = semaphoreProperties.externalSemaphoreFeatures;
        context->syncFdImport = (features & VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) != 0;
        context->syncFdExport = (features & VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT) != 0;
        if (context->syncFdImport or context->syncFdExport) {
            extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);
        }
    }
    // The transfer queue needs timeline semaphores to wait across queues.
    auto const transfer = context->timelineSemaphores ? transferQueueOf(physical, queueFamily) : std::nullopt;
    float const priorities[] = {1.0f, 1.0f};
    auto queueInfos = std::vector<VkDeviceQueueCreateInfo>{};
    auto queueInfo = vkStruct<VkDeviceQueueCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
    queueInfo.queueFamilyIndex = queueFamily;
    queueInfo.queueCount = transfer and transfer->first == queueFamily ? 2 : 1;
    queueInfo.pQueuePriorities = priorities;
    queueInfos.push_back(queueInfo);
    if (transfer and transfer->first != queueFamily) {
        queueInfo.queueFamilyIndex = transfer->first;
        queueInfo.queueCount = 1;
        queueInfos.push_back(queueInfo);
    }
    info.queueCreateInfoCount = static_cast<std::uint32_t>(queueInfos.size());
    info.pQueueCreateInfos = queueInfos.data();
    info.pEnabledFeatures = &features;
    info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();
    check(vkCreateDevice(physical, &info, nullptr, &device.handle), "vkCreateDevice");
    vkGetDeviceQueue(device.handle, queueFamily, 0, &queue);
    if (transfer) {
        context->transferFamily = transfer->first;
        context->transferIndex = transfer->second;
        vkGetDeviceQueue(device.handle, transfer->first, transfer->second, &context->transferQueue);
    }
    if (context->syncFdImport or context->syncFdExport) {
        getSemaphoreFd = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(vkGetDeviceProcAddr(device.handle, "vkGetSemaphoreFdKHR"));
        importSemaphoreFd =
            reinterpret_cast<PFN_vkImportSemaphoreFdKHR>(vkGetDeviceProcAddr(device.handle, "vkImportSemaphoreFdKHR"));
        context->syncFdImport = context->syncFdImport and importSemaphoreFd != nullptr;
        context->syncFdExport = context->syncFdExport and getSemaphoreFd != nullptr;
    }
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
    info.usage = kBufferUsage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    shareAcrossQueues(info);
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

StagingChunk VulkanDevice::makeStaging(VkDeviceSize const capacity) {
    auto chunk = StagingChunk{};
    auto info = vkStruct<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    info.size = capacity;
    info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto buffer = VkBuffer{};
    check(vkCreateBuffer(device.handle, &info, nullptr, &buffer), "vkCreateBuffer");
    chunk.buffer = DeviceObject<VkBuffer>{device.handle, buffer, vkDestroyBuffer};
    auto requirements = VkMemoryRequirements{};
    vkGetBufferMemoryRequirements(device.handle, buffer, &requirements);
    chunk.memory = allocate(requirements, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check(vkBindBufferMemory(device.handle, buffer, chunk.memory.get(), 0), "vkBindBufferMemory");
    void *mapped = nullptr;
    check(vkMapMemory(device.handle, chunk.memory.get(), 0, capacity, 0, &mapped), "vkMapMemory");
    chunk.mapped = static_cast<std::byte *>(mapped);
    chunk.capacity = capacity;
    return chunk;
}

DeviceObject<VkDescriptorPool> VulkanDevice::makeDescriptorPool() {
    VkDescriptorPoolSize const sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 512},
                                          {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 512},
                                          {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 512}};
    auto info = vkStruct<VkDescriptorPoolCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
    info.maxSets = 128;
    info.poolSizeCount = 3;
    info.pPoolSizes = sizes;
    auto pool = VkDescriptorPool{};
    check(vkCreateDescriptorPool(device.handle, &info, nullptr, &pool), "vkCreateDescriptorPool");
    return DeviceObject<VkDescriptorPool>{device.handle, pool, vkDestroyDescriptorPool};
}

Batch &VulkanDevice::recordingBatch() {
    if (recordingNow != nullptr) {
        return *recordingNow;
    }
    retireCompleted();
    if (idleBatches.empty()) {
        auto batch = std::make_unique<Batch>();
        auto info = vkStruct<VkCommandBufferAllocateInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
        info.commandPool = commandPool.get();
        info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        info.commandBufferCount = 1;
        check(vkAllocateCommandBuffers(device.handle, &info, &batch->commandBuffer), "vkAllocateCommandBuffers");
        idleBatches.push_back(std::move(batch));
    }
    recordingNow = std::move(idleBatches.back());
    idleBatches.pop_back();
    recordingNow->serial = nextSerial++;
    recordingNow->submission = nextSubmission();
    check(vkResetCommandBuffer(recordingNow->commandBuffer, 0), "vkResetCommandBuffer");
    auto begin = vkStruct<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(recordingNow->commandBuffer, &begin), "vkBeginCommandBuffer");
    return *recordingNow;
}

VkCommandBuffer VulkanDevice::commandFor(std::vector<detail::ResourceAccess> const &accesses) {
    auto const cmd = recording();
    if (hazards.needsBarrier(accesses)) {
        fullBarrier(cmd);
    }
    return cmd;
}

// Sends the recording batch with a fence of its own. The hazard tracker
// carries over: a barrier in a later batch also orders earlier batches.
Submission VulkanDevice::submitBatch(DeviceObject<VkSemaphore> signal) {
    auto const signalHandle = signal.get();
    if (recordingNow == nullptr) {
        if (pendingWaits.empty() and pendingTimelineWaits.empty() and signalHandle == VK_NULL_HANDLE) {
            return inFlight.empty() ? Submission{} : Submission{inFlight.back()->completion};
        }
        recordingBatch(); // an empty batch carries the semaphores
    }
    auto &batch = *recordingNow;
    // Imported buffers go back to the device they came from.
    for (auto const buffer : batch.imported) {
        auto barrier = vkStruct<VkBufferMemoryBarrier>(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER);
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = 0;
        barrier.srcQueueFamilyIndex = queueFamily;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        barrier.buffer = buffer;
        barrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(batch.commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             0, 0, nullptr, 1, &barrier, 0, nullptr);
    }
    batch.imported.clear();
    hostVisibleBarrier(batch.commandBuffer);
    check(vkEndCommandBuffer(batch.commandBuffer), "vkEndCommandBuffer");
    auto fenceInfo = vkStruct<VkFenceCreateInfo>(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    auto rawFence = VkFence{};
    check(vkCreateFence(device.handle, &fenceInfo, nullptr, &rawFence), "vkCreateFence");
    auto fence = DeviceObject<VkFence>{device.handle, rawFence, vkDestroyFence};
    // Binary semaphores first (their values are ignored), then the timelines.
    auto waits = std::vector<VkSemaphore>{};
    auto waitValues = std::vector<std::uint64_t>{};
    for (auto const &semaphore : pendingWaits) {
        waits.push_back(semaphore.get());
        waitValues.push_back(0);
    }
    for (auto const &[semaphore, value] : pendingTimelineWaits) {
        waits.push_back(semaphore);
        waitValues.push_back(value);
    }
    auto signals = std::vector<VkSemaphore>{};
    auto signalValues = std::vector<std::uint64_t>{};
    if (signalHandle != VK_NULL_HANDLE) {
        signals.push_back(signalHandle);
        signalValues.push_back(0);
    }
    if (mainTimeline.get() != VK_NULL_HANDLE) {
        signals.push_back(mainTimeline.get());
        signalValues.push_back(batch.serial);
    }
    auto const stages = std::vector<VkPipelineStageFlags>(waits.size(), VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    auto timelines = vkStruct<VkTimelineSemaphoreSubmitInfo>(VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO);
    timelines.waitSemaphoreValueCount = static_cast<std::uint32_t>(waitValues.size());
    timelines.pWaitSemaphoreValues = waitValues.data();
    timelines.signalSemaphoreValueCount = static_cast<std::uint32_t>(signalValues.size());
    timelines.pSignalSemaphoreValues = signalValues.data();
    auto info = vkStruct<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
    info.pNext = mainTimeline.get() != VK_NULL_HANDLE ? &timelines : nullptr;
    info.waitSemaphoreCount = static_cast<std::uint32_t>(waits.size());
    info.pWaitSemaphores = waits.data();
    info.pWaitDstStageMask = stages.data();
    info.commandBufferCount = 1;
    info.pCommandBuffers = &batch.commandBuffer;
    info.signalSemaphoreCount = static_cast<std::uint32_t>(signals.size());
    info.pSignalSemaphores = signals.data();
    check(vkQueueSubmit(queue, 1, &info, rawFence), "vkQueueSubmit");
    for (auto &semaphore : pendingWaits) {
        batch.semaphores.push_back(std::move(semaphore));
    }
    pendingWaits.clear();
    pendingTimelineWaits.clear();
    if (signalHandle != VK_NULL_HANDLE) {
        batch.semaphores.push_back(std::move(signal));
    }
    batch.completion = std::make_shared<FenceCompletion>(context, std::move(fence), mainTimeline.get(), batch.serial);
    auto submission = Submission{batch.completion};
    inFlight.push_back(std::move(recordingNow));
    return submission;
}

void VulkanDevice::flush() {
    if (transferQueue != nullptr) {
        transferQueue->flush();
    }
    submitBatch();
    for (auto const &batch : inFlight) {
        batch->completion->wait();
    }
    retireCompleted();
}

// Waits for a transfer submission go into the next batch; work recorded
// before this call is sent first, so it does not wait.
void VulkanDevice::doWaitForSubmission(Submission const &submission) {
    auto const *completion = dynamic_cast<FenceCompletion const *>(completionOf(submission));
    if (completion == nullptr or transferQueue == nullptr or completion->timeline == mainTimeline.get()) {
        return; // this timeline's own work is in order already
    }
    submitBatch();
    pendingTimelineWaits.emplace_back(completion->timeline, completion->value);
}

// Recycles the batches the device has finished, in order, and destroys what
// only they still used.
void VulkanDevice::retireCompleted() {
    while (not inFlight.empty() and inFlight.front()->completion->isComplete()) {
        auto &batch = *inFlight.front();
        readTimestamps(batch);
        for (auto &pool : batch.descriptorPools) {
            check(vkResetDescriptorPool(device.handle, pool.get(), 0), "vkResetDescriptorPool");
        }
        batch.descriptorPool = 0;
        for (auto &chunk : batch.staging) {
            chunk.used = 0;
        }
        batch.semaphores.clear();
        completedSerial = batch.serial;
        batch.completion.reset();
        idleBatches.push_back(std::move(inFlight.front()));
        inFlight.pop_front();
    }
    auto const transfersDone = transferQueue != nullptr ? transferQueue->retire() : std::uint64_t{0};
    auto const done = std::remove_if(graveyard.begin(), graveyard.end(), [&](Grave const &grave) {
        return grave.serial <= completedSerial and grave.transferSerial <= transfersDone;
    });
    graveyard.erase(done, graveyard.end());
}

// Keeps `object` until every batch of either queue that may use it has
// completed.
template <class T>
void VulkanDevice::bury(T &&object) {
    auto const serial = recordingNow != nullptr ? recordingNow->serial : nextSerial - 1;
    auto const transferSerial = transferQueue != nullptr ? transferQueue->latestSerial() : std::uint64_t{0};
    auto const transfersDone = transferQueue != nullptr ? transferQueue->completed() : std::uint64_t{0};
    if (serial > completedSerial or transferSerial > transfersDone) {
        graveyard.push_back(
            Grave{serial, transferSerial, std::make_shared<std::decay_t<T>>(std::forward<T>(object))});
    }
}

DeviceObject<VkSemaphore> VulkanDevice::makeSemaphore(bool const exportable) {
    auto exportInfo = vkStruct<VkExportSemaphoreCreateInfo>(VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO);
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    auto info = vkStruct<VkSemaphoreCreateInfo>(VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
    info.pNext = exportable ? &exportInfo : nullptr;
    auto semaphore = VkSemaphore{};
    check(vkCreateSemaphore(device.handle, &info, nullptr, &semaphore), "vkCreateSemaphore");
    return DeviceObject<VkSemaphore>{device.handle, semaphore, vkDestroySemaphore};
}

// The queue waits for the sync file before the next submission; work
// queued before this call is submitted first, so it does not wait.
void VulkanDevice::doWaitFor(SyncFence fence) {
    if (fence.empty()) {
        return;
    }
    if (not context->syncFdImport) {
        fence.wait();
        return;
    }
    submitBatch();
    auto semaphore = makeSemaphore(false);
    auto import = vkStruct<VkImportSemaphoreFdInfoKHR>(VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR);
    import.semaphore = semaphore.get();
    import.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
    import.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    import.fd = fence.get();
    if (importSemaphoreFd(device.handle, &import) != VK_SUCCESS) {
        fence.wait();
        return;
    }
    fence.release(); // Vulkan owns the descriptor now
    pendingWaits.push_back(std::move(semaphore));
}

FencedSubmission VulkanDevice::doSubmitWithFence() {
    if (not context->syncFdExport) {
        auto submission = submitBatch();
        submission.wait();
        return FencedSubmission{std::move(submission), SyncFence{}};
    }
    auto semaphore = makeSemaphore(true);
    auto const handle = semaphore.get();
    auto submission = submitBatch(std::move(semaphore));
    auto info = vkStruct<VkSemaphoreGetFdInfoKHR>(VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR);
    info.semaphore = handle;
    info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    auto fd = -1;
    check(getSemaphoreFd(device.handle, &info, &fd), "vkGetSemaphoreFdKHR");
    return FencedSubmission{std::move(submission), SyncFence{fd}}; // -1 when the work has already finished
}

// Batches complete in order, so waiting for those of submissions up to
// `last` leaves later batches running.
std::vector<KernelTiming> VulkanDevice::doTakeTimings(std::uint64_t const last) {
    for (auto const &batch : inFlight) {
        if (batch->submission <= last) {
            batch->completion->wait();
        }
    }
    retireCompleted();
    return takeThrough(timings, last);
}

std::uint32_t VulkanDevice::reserveTimestamps(VkCommandBuffer const cmd) {
    auto &batch = recordingBatch();
    if (batch.queryPool.get() == VK_NULL_HANDLE) {
        auto info = vkStruct<VkQueryPoolCreateInfo>(VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO);
        info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        info.queryCount = kQueryCapacity;
        auto pool = VkQueryPool{};
        check(vkCreateQueryPool(device.handle, &info, nullptr, &pool), "vkCreateQueryPool");
        batch.queryPool = DeviceObject<VkQueryPool>{device.handle, pool, vkDestroyQueryPool};
    }
    if (batch.queriesUsed == 0) {
        vkCmdResetQueryPool(cmd, batch.queryPool.get(), 0, kQueryCapacity);
    }
    auto const first = batch.queriesUsed;
    batch.queriesUsed += 2;
    return first;
}

// Converts the timestamps of a batch that has completed.
void VulkanDevice::readTimestamps(Batch &batch) {
    if (batch.queriesUsed == 0) {
        return;
    }
    auto ticks = std::vector<std::uint64_t>(batch.queriesUsed);
    check(vkGetQueryPoolResults(device.handle, batch.queryPool.get(), 0, batch.queriesUsed,
                                ticks.size() * sizeof(std::uint64_t), ticks.data(), sizeof(std::uint64_t),
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
          "vkGetQueryPoolResults");
    auto const mask = timestampBits >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << timestampBits) - 1;
    auto const nanosecondsPerTick = static_cast<double>(properties.limits.timestampPeriod);
    for (auto &pending : batch.pendingTimings) {
        auto const elapsed = (ticks[pending.firstQuery + 1] - ticks[pending.firstQuery]) & mask;
        timings.push_back(KernelTiming{std::move(pending.program), pending.workGroups,
                                       static_cast<double>(elapsed) * nanosecondsPerTick * 1e-6, pending.indirect,
                                       batch.submission});
    }
    batch.pendingTimings.clear();
    batch.queriesUsed = 0;
}

// `size` staging bytes no other command of the recording batch uses.
StagingSlice VulkanDevice::stagingSlice(VkDeviceSize const size) {
    auto &batch = recordingBatch();
    for (auto &chunk : batch.staging) {
        auto const offset = (chunk.used + 15) / 16 * 16;
        if (offset + size <= chunk.capacity) {
            chunk.used = offset + size;
            return StagingSlice{chunk.buffer.get(), offset, chunk.mapped + offset};
        }
    }
    auto const previous = batch.staging.empty() ? VkDeviceSize{0} : batch.staging.back().capacity;
    batch.staging.push_back(makeStaging(std::max({size, VkDeviceSize{1} << 20, 2 * previous})));
    auto &chunk = batch.staging.back();
    chunk.used = size;
    return StagingSlice{chunk.buffer.get(), 0, chunk.mapped};
}

VkDescriptorSet VulkanDevice::allocateSet(VkDescriptorSetLayout const layout) {
    auto &batch = recordingBatch();
    for (;;) {
        if (batch.descriptorPool == batch.descriptorPools.size()) {
            batch.descriptorPools.push_back(makeDescriptorPool());
        }
        auto info = vkStruct<VkDescriptorSetAllocateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
        info.descriptorPool = batch.descriptorPools[batch.descriptorPool].get();
        info.descriptorSetCount = 1;
        info.pSetLayouts = &layout;
        auto set = VkDescriptorSet{};
        auto const result = vkAllocateDescriptorSets(device.handle, &info, &set);
        if (result == VK_ERROR_OUT_OF_POOL_MEMORY or result == VK_ERROR_FRAGMENTED_POOL) {
            ++batch.descriptorPool;
            continue;
        }
        check(result, "vkAllocateDescriptorSets");
        return set;
    }
}

// Releases mirrors of destroyed host resources once queued work is done
// with them.
void VulkanDevice::collectGarbage() {
    retireCompleted();
    for (auto it = heaps.begin(); it != heaps.end();) {
        if (it->second.owner.expired()) {
            bury(std::move(it->second));
            it = heaps.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = buffers.begin(); it != buffers.end();) {
        if (it->second.owner.expired()) {
            bury(std::move(it->second));
            it = buffers.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = images.begin(); it != images.end();) {
        if (it->second.owner.expired()) {
            bury(std::move(it->second));
            it = images.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = externalImages.begin(); it != externalImages.end();) {
        if (it->second.owner.expired()) {
            bury(std::move(it->second));
            it = externalImages.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = externals.begin(); it != externals.end();) {
        if (not it->second.owner.expired()) {
            ++it;
            continue;
        }
        // Pipelines bake the frame's sampler into their layout; drop them too.
        auto const token = "|" + std::to_string(handleKey(it->second.sampler.get()));
        for (auto pipeline = pipelines.begin(); pipeline != pipelines.end();) {
            if (pipeline->first.find(token) != std::string::npos) {
                bury(std::move(pipeline->second));
                pipeline = pipelines.erase(pipeline);
            } else {
                ++pipeline;
            }
        }
        bury(std::move(it->second));
        it = externals.erase(it);
    }
}

void VulkanDevice::doUploadBuffer(ResourceIdentity const &identity, ConstByteSpan const bytes, ByteRange range) {
    if (bytes.empty()) {
        throw DeviceError{"uploadBuffer: Vulkan buffers cannot be empty"};
    }
    collectGarbage();
    auto &mirror = buffers[identity.key()];
    if (mirror.mapped and mirror.size == bytes.size() and mirror.mapped->mapped == bytes.data()) {
        return; // createBuffer() memory: the device reads it in place
    }
    if (mirror.size != bytes.size()) {
        range = ByteRange{0, bytes.size()}; // a new device buffer starts with the whole contents
        if (mirror.size != 0) {
            bury(std::move(mirror)); // queued commands may still use the old buffer
        }
        mirror = BufferMirror{};
        mirror.owner = identity.watch();
        auto info = vkStruct<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
        info.size = bytes.size();
        info.usage = kBufferUsage;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        shareAcrossQueues(info);
        auto buffer = VkBuffer{};
        check(vkCreateBuffer(device.handle, &info, nullptr, &buffer), "vkCreateBuffer");
        mirror.buffer = DeviceObject<VkBuffer>{device.handle, buffer, vkDestroyBuffer};
        auto requirements = VkMemoryRequirements{};
        vkGetBufferMemoryRequirements(device.handle, buffer, &requirements);
        mirror.memory = allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkBindBufferMemory(device.handle, buffer, mirror.memory.get(), 0), "vkBindBufferMemory");
        mirror.size = bytes.size();
    }
    if (range.size == 0) {
        return;
    }
    auto const slice = stagingSlice(range.size);
    std::memcpy(slice.mapped, bytes.data() + range.offset, range.size);
    auto const region = VkBufferCopy{slice.offset, range.offset, range.size};
    auto const cmd = commandFor({detail::ResourceAccess{identity.key(), range.offset, range.size, false, true}});
    vkCmdCopyBuffer(cmd, slice.buffer, mirror.handle(), 1, &region);
}

void VulkanDevice::doDownloadBuffer(ResourceIdentity const &identity, ByteSpan const bytes, ByteRange const range) {
    auto const it = buffers.find(identity.key());
    if (it == buffers.end() or it->second.size != bytes.size()) {
        throw DeviceError{"downloadBuffer: the buffer was never uploaded to this device (or changed size)"};
    }
    if (it->second.mapped and it->second.mapped->mapped == bytes.data()) {
        flush(); // createBuffer() memory: GPU writes are visible once the batch completes
        return;
    }
    if (range.size == 0) {
        return;
    }
    auto const slice = stagingSlice(range.size);
    auto const region = VkBufferCopy{range.offset, slice.offset, range.size};
    auto const cmd = commandFor({detail::ResourceAccess{identity.key(), range.offset, range.size, true, false}});
    vkCmdCopyBuffer(cmd, it->second.handle(), slice.buffer, 1, &region);
    hostReadBarrier(cmd);
    flush(); // the slice keeps its bytes until the next batch is recorded
    std::memcpy(bytes.data() + range.offset, slice.mapped, range.size);
}

VkImageCreateInfo VulkanDevice::imageInfo(ImageFormat const &imageFormat, std::optional<CompressedFormat> const compressed,
                                          Dim const dim, ivec3 const extent, std::uint32_t const levels) const {
    auto const format = compressed ? vkCompressedFormat(*compressed) : vkFormatOf(imageFormat);
    // The same image may be written as a storage image and read as a
    // sampled texture later, so request every usage the format supports.
    // Compressed images are only sampled.
    auto const features = formatFeatures(format);
    auto usage = VkImageUsageFlags{VK_IMAGE_USAGE_TRANSFER_DST_BIT};
    if (not compressed) {
        usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if ((features & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0) {
            usage |= VK_IMAGE_USAGE_STORAGE_BIT;
        }
    }
    if ((features & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0) {
        usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
    }
    if ((usage & (VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT)) == 0) {
        auto const name = compressed ? std::string{compressedFormatInfo(*compressed).name} : glslFormatQualifier(imageFormat);
        throw DeviceError{"image format " + name + " can be neither stored nor sampled on " + properties.deviceName};
    }
    auto info = vkStruct<VkImageCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);
    info.imageType = dim == Dim::D3 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = levelExtent(dim, extent, 0);
    info.mipLevels = levels;
    info.arrayLayers = layerCount(dim, extent);
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    shareAcrossQueues(info);
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    return info;
}

void VulkanDevice::allocateImage(ImageMirror &mirror, ResourceIdentity const &identity, ImageFormat const &imageFormat,
                                 std::optional<CompressedFormat> const compressed, Dim const dim, ivec3 const extent,
                                 std::uint32_t const levels, std::optional<HeapRange> const &placement) {
    auto const info = imageInfo(imageFormat, compressed, dim, extent, levels);
    auto const layers = info.arrayLayers;
    if (mirror.image.get() != VK_NULL_HANDLE) {
        bury(std::move(mirror)); // queued commands may still use the old image
    }
    mirror = ImageMirror{};
    mirror.usage = info.usage;
    mirror.owner = identity.watch();
    mirror.format = imageFormat;
    mirror.compressed = compressed;
    mirror.dim = dim;
    mirror.extent = extent;
    mirror.levels = levels;
    auto image = VkImage{};
    check(vkCreateImage(device.handle, &info, nullptr, &image), "vkCreateImage");
    mirror.image = DeviceObject<VkImage>{device.handle, image, vkDestroyImage};
    auto requirements = VkMemoryRequirements{};
    vkGetImageMemoryRequirements(device.handle, image, &requirements);
    if (placement) {
        mirror.heap = heapMemory(*placement, requirements, "createImage");
        check(vkBindImageMemory(device.handle, image, mirror.heap->get(), placement->offset), "vkBindImageMemory");
        mirror.view = createView(mirror, 0, levels);
        aliases.place(identity.key(), identity.watch(),
                      detail::AliasTracker::Range{placement->heap->key(), placement->offset,
                                                  static_cast<std::size_t>(requirements.size)});
        return;
    }
    mirror.memory = allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkBindImageMemory(device.handle, image, mirror.memory.get(), 0), "vkBindImageMemory");
    mirror.view = createView(mirror, 0, levels);
    // Nothing used the new image before, so this is its only barrier.
    auto barrier = vkStruct<VkImageMemoryBarrier>(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER);
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, layers};
    vkCmdPipelineBarrier(recording(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
}

bool VulkanDevice::doCreateImage(ImageCreation const &request) {
    collectGarbage();
    allocateImage(images[request.identity->key()], *request.identity, request.format, request.compressed, request.dim,
                  request.extent, request.levels, request.placement);
    return false;
}

PlacementSize VulkanDevice::placementOf(VkMemoryRequirements const &requirements) const {
    // Buffers and optimally tiled images may share a heap: keeping every
    // resource on whole granules keeps them off each other's pages.
    auto const granularity = std::max<VkDeviceSize>(properties.limits.bufferImageGranularity, 1);
    auto const bytes = (requirements.size + granularity - 1) / granularity * granularity;
    return PlacementSize{static_cast<std::size_t>(bytes),
                         static_cast<std::size_t>(std::max(requirements.alignment, granularity))};
}

PlacementSize VulkanDevice::doBufferPlacement(std::size_t const bytes) {
    auto info = vkStruct<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    info.size = bytes;
    info.usage = kBufferUsage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    shareAcrossQueues(info);
    auto buffer = VkBuffer{};
    check(vkCreateBuffer(device.handle, &info, nullptr, &buffer), "vkCreateBuffer");
    auto const probe = DeviceObject<VkBuffer>{device.handle, buffer, vkDestroyBuffer};
    auto requirements = VkMemoryRequirements{};
    vkGetBufferMemoryRequirements(device.handle, probe.get(), &requirements);
    return placementOf(requirements);
}

PlacementSize VulkanDevice::doImagePlacement(ImageCreation const &request) {
    auto const info = imageInfo(request.format, request.compressed, request.dim, request.extent, request.levels);
    auto image = VkImage{};
    check(vkCreateImage(device.handle, &info, nullptr, &image), "vkCreateImage");
    auto const probe = DeviceObject<VkImage>{device.handle, image, vkDestroyImage};
    auto requirements = VkMemoryRequirements{};
    vkGetImageMemoryRequirements(device.handle, probe.get(), &requirements);
    return placementOf(requirements);
}

// Device-local memory that both buffers and (RGBA16F, the common case of
// transient images) optimally tiled images can live in.
bool VulkanDevice::doCreateHeap(HeapCreation const &request) {
    collectGarbage();
    auto bufferInfo = vkStruct<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    bufferInfo.size = 256;
    bufferInfo.usage = kBufferUsage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    shareAcrossQueues(bufferInfo);
    auto buffer = VkBuffer{};
    check(vkCreateBuffer(device.handle, &bufferInfo, nullptr, &buffer), "vkCreateBuffer");
    auto const bufferProbe = DeviceObject<VkBuffer>{device.handle, buffer, vkDestroyBuffer};
    auto bufferNeeds = VkMemoryRequirements{};
    vkGetBufferMemoryRequirements(device.handle, bufferProbe.get(), &bufferNeeds);
    auto const imageCreate = imageInfo(ImageFormat{Encoding::Float, false, 16, 4}, std::nullopt, Dim::D2, ivec3{64, 64, 1}, 1);
    auto image = VkImage{};
    check(vkCreateImage(device.handle, &imageCreate, nullptr, &image), "vkCreateImage");
    auto const imageProbe = DeviceObject<VkImage>{device.handle, image, vkDestroyImage};
    auto imageNeeds = VkMemoryRequirements{};
    vkGetImageMemoryRequirements(device.handle, imageProbe.get(), &imageNeeds);
    auto const both = bufferNeeds.memoryTypeBits & imageNeeds.memoryTypeBits;
    auto typeBits = bufferNeeds.memoryTypeBits;
    for (std::uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
        if ((both & (1u << i)) != 0
            and (memoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) {
            typeBits = both;
            break;
        }
    }
    auto &heap = heaps[request.identity->key()];
    heap.owner = request.identity->watch();
    heap.memoryType = memoryType(typeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    heap.size = request.bytes;
    auto info = vkStruct<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    info.allocationSize = request.bytes;
    info.memoryTypeIndex = heap.memoryType;
    auto memory = VkDeviceMemory{};
    check(vkAllocateMemory(device.handle, &info, nullptr, &memory), "vkAllocateMemory (heap)");
    heap.memory = std::make_shared<DeviceObject<VkDeviceMemory>>(device.handle, memory, vkFreeMemory);
    return false;
}

std::shared_ptr<DeviceObject<VkDeviceMemory>> VulkanDevice::heapMemory(HeapRange const &placement,
                                                                       VkMemoryRequirements const &requirements,
                                                                       char const *what) const {
    auto const &heap = heaps.at(placement.heap->key());
    if ((requirements.memoryTypeBits & (1u << heap.memoryType)) == 0) {
        throw DeviceError{std::string{what} + ": this resource cannot live in the memory of a heap on "
                          + properties.deviceName + "; give it memory of its own"};
    }
    if (placement.offset % requirements.alignment != 0 or placement.offset + requirements.size > heap.size) {
        throw DeviceError{std::string{what} + ": the placement is misaligned or leaves the heap"};
    }
    return heap.memory;
}

bool VulkanDevice::doCreateDeviceBuffer(BufferCreation const &request) {
    collectGarbage();
    auto &mirror = buffers[request.identity->key()];
    mirror = BufferMirror{};
    mirror.owner = request.identity->watch();
    auto info = vkStruct<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    info.size = request.bytes;
    info.usage = kBufferUsage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    shareAcrossQueues(info);
    auto buffer = VkBuffer{};
    check(vkCreateBuffer(device.handle, &info, nullptr, &buffer), "vkCreateBuffer");
    mirror.buffer = DeviceObject<VkBuffer>{device.handle, buffer, vkDestroyBuffer};
    auto requirements = VkMemoryRequirements{};
    vkGetBufferMemoryRequirements(device.handle, buffer, &requirements);
    if (request.placement) {
        mirror.heap = heapMemory(*request.placement, requirements, "createDeviceBuffer");
        check(vkBindBufferMemory(device.handle, buffer, mirror.heap->get(), request.placement->offset),
              "vkBindBufferMemory");
        aliases.place(request.identity->key(), request.identity->watch(),
                      detail::AliasTracker::Range{request.placement->heap->key(), request.placement->offset,
                                                  request.bytes});
    } else {
        mirror.memory = allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkBindBufferMemory(device.handle, buffer, mirror.memory.get(), 0), "vkBindBufferMemory");
    }
    mirror.size = request.bytes;
    return false;
}

void VulkanDevice::doImportBuffer(BufferImport const &request) {
    if (not context->memoryImport) {
        throw DeviceError{"importBuffer: " + std::string{properties.deviceName}
                          + " cannot import memory (needs Vulkan 1.1 and VK_KHR_external_memory_fd)"};
    }
    collectGarbage();
    auto &memory = *request.memory;
    auto externalProperties = vkStruct<VkExternalBufferProperties>(VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES);
    auto externalInfo = vkStruct<VkPhysicalDeviceExternalBufferInfo>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO);
    externalInfo.usage = kBufferUsage;
    externalInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    vkGetPhysicalDeviceExternalBufferProperties(physical, &externalInfo, &externalProperties);
    auto const features = externalProperties.externalMemoryProperties.externalMemoryFeatures;
    if ((features & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) == 0
        or (features & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT) != 0) {
        throw DeviceError{"importBuffer: " + std::string{properties.deviceName}
                          + " cannot import opaque file descriptors as buffers"};
    }
    if (memory.memoryType >= memoryProperties.memoryTypeCount) {
        throw DeviceError{"importBuffer: memory type " + std::to_string(memory.memoryType) + " does not exist on "
                          + properties.deviceName};
    }
    auto mirror = BufferMirror{};
    mirror.owner = request.identity->watch();
    mirror.imported = true;
    auto externalBuffer = vkStruct<VkExternalMemoryBufferCreateInfo>(VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO);
    externalBuffer.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    auto info = vkStruct<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    info.pNext = &externalBuffer;
    info.size = request.bytes;
    info.usage = kBufferUsage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto buffer = VkBuffer{};
    check(vkCreateBuffer(device.handle, &info, nullptr, &buffer), "vkCreateBuffer (imported)");
    mirror.buffer = DeviceObject<VkBuffer>{device.handle, buffer, vkDestroyBuffer};
    auto requirements = VkMemoryRequirements{};
    vkGetBufferMemoryRequirements(device.handle, buffer, &requirements);
    if ((requirements.memoryTypeBits & (1u << memory.memoryType)) == 0 or memory.offset % requirements.alignment != 0
        or memory.offset + requirements.size > memory.allocationBytes) {
        throw DeviceError{"importBuffer: a " + std::to_string(request.bytes) + "-byte buffer cannot live at byte "
                          + std::to_string(memory.offset) + " of memory type " + std::to_string(memory.memoryType)};
    }
    auto import = vkStruct<VkImportMemoryFdInfoKHR>(VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR);
    import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
    import.fd = memory.fd.get();
    auto allocateInfo = vkStruct<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    allocateInfo.pNext = &import;
    allocateInfo.allocationSize = memory.allocationBytes;
    allocateInfo.memoryTypeIndex = memory.memoryType;
    auto raw = VkDeviceMemory{};
    check(vkAllocateMemory(device.handle, &allocateInfo, nullptr, &raw), "vkAllocateMemory (import)");
    memory.fd.release(); // Vulkan owns the descriptor now
    mirror.memory = DeviceObject<VkDeviceMemory>{device.handle, raw, vkFreeMemory};
    check(vkBindBufferMemory(device.handle, buffer, raw, memory.offset), "vkBindBufferMemory (imported)");
    mirror.size = request.bytes;
    buffers[request.identity->key()] = std::move(mirror);
}

void VulkanDevice::claim(VkCommandBuffer const cmd, void const *const key) {
    if (auto const buffer = buffers.find(key); buffer != buffers.end() and buffer->second.imported) {
        auto &acquired = recordingBatch().imported;
        auto const handle = buffer->second.handle();
        if (std::find(acquired.begin(), acquired.end(), handle) == acquired.end()) {
            auto barrier = vkStruct<VkBufferMemoryBarrier>(VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER);
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
            barrier.dstQueueFamilyIndex = queueFamily;
            barrier.buffer = handle;
            barrier.size = VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                                 nullptr, 1, &barrier, 0, nullptr);
            acquired.push_back(handle);
        }
        return;
    }
    if (not aliases.claim(key)) {
        return;
    }
    auto const it = images.find(key);
    if (it == images.end()) {
        return; // buffers need no more than the barrier their accesses already got
    }
    auto const &mirror = it->second;
    auto barrier = vkStruct<VkImageMemoryBarrier>(VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER);
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = mirror.image.get();
    barrier.subresourceRange =
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, mirror.levels, 0, layerCount(mirror.dim, mirror.extent)};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &barrier);
}

void VulkanDevice::doCopyBuffer(BufferCopy const &copy) {
    auto const source = buffers.find(copy.source->key());
    auto const target = buffers.find(copy.target->key());
    if (source == buffers.end() or target == buffers.end()) {
        throw DeviceError{"copyBuffer: a buffer was never uploaded to this device"};
    }
    auto const cmd = commandFor({access(copy.source->key(), copy.sourceOffset, copy.bytes, true, false),
                                 access(copy.target->key(), copy.targetOffset, copy.bytes, false, true)});
    claim(cmd, copy.source->key());
    claim(cmd, copy.target->key());
    auto const region = VkBufferCopy{copy.sourceOffset, copy.targetOffset, copy.bytes};
    vkCmdCopyBuffer(cmd, source->second.handle(), target->second.handle(), 1, &region);
}

VkBufferImageCopy VulkanDevice::regionOf(BufferImageCopy const &copy) {
    auto region = VkBufferImageCopy{};
    region.bufferOffset = copy.offset;
    if (copy.dim == Dim::D3) {
        region.imageSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, copy.level, 0, 1};
        region.imageOffset = VkOffset3D{0, 0, static_cast<std::int32_t>(copy.firstLayer)};
        region.imageExtent = VkExtent3D{static_cast<std::uint32_t>(copy.extent.x),
                                        static_cast<std::uint32_t>(copy.extent.y), copy.layers};
    } else {
        region.imageSubresource =
            VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, copy.level, copy.firstLayer, copy.layers};
        region.imageExtent = VkExtent3D{static_cast<std::uint32_t>(copy.extent.x),
                                        static_cast<std::uint32_t>(copy.extent.y), 1};
    }
    return region;
}

void VulkanDevice::doCopyBufferToImage(BufferImageCopy const &copy) {
    auto const buffer = buffers.find(copy.buffer->key());
    if (buffer == buffers.end() or buffer->second.size < copy.offset + copy.bytes) {
        throw DeviceError{"copyBufferToImage: the buffer was never uploaded to this device (or changed size)"};
    }
    auto const &mirror = images.at(copy.image->key());
    auto const region = regionOf(copy);
    auto const cmd = commandFor({access(copy.buffer->key(), copy.offset, copy.bytes, true, false),
                                 access(copy.image->key(), 0, 0, false, true)});
    claim(cmd, copy.buffer->key());
    claim(cmd, copy.image->key());
    vkCmdCopyBufferToImage(cmd, buffer->second.handle(), mirror.image.get(), VK_IMAGE_LAYOUT_GENERAL, 1, &region);
}

void VulkanDevice::doUploadImage(ImageTransfer const &transfer) {
    collectGarbage();
    auto &mirror = images[transfer.identity->key()];
    auto const isNew = mirror.image.get() == VK_NULL_HANDLE or not(mirror.format == transfer.format)
                    or not(mirror.extent == transfer.extent) or mirror.levels != transfer.levels;
    if (isNew) {
        allocateImage(mirror, *transfer.identity, transfer.format, std::nullopt, transfer.dim, transfer.extent,
                      transfer.levels);
    }
    auto const first = isNew ? 0u : transfer.firstLevel; // a new image starts with every level
    auto const count = isNew ? transfer.levels : transfer.levelCount;
    auto offsets = std::vector<VkDeviceSize>{};
    auto total = VkDeviceSize{0};
    for (auto level = first; level < first + count; ++level) {
        offsets.push_back(total);
        total += (levelBytes(transfer, level) + 15) / 16 * 16;
    }
    auto const slice = stagingSlice(total);
    auto regions = std::vector<VkBufferImageCopy>{};
    for (auto level = first; level < first + count; ++level) {
        auto const offset = offsets[level - first];
        transfer.writeTexels(level, ByteSpan{slice.mapped + offset, levelBytes(transfer, level)});
        regions.push_back(imageCopy(transfer, level, slice.offset + offset));
    }
    auto const cmd = commandFor({detail::ResourceAccess{transfer.identity->key(), 0, 0, false, true}});
    vkCmdCopyBufferToImage(cmd, slice.buffer, mirror.image.get(), VK_IMAGE_LAYOUT_GENERAL,
                           static_cast<std::uint32_t>(regions.size()), regions.data());
}

void VulkanDevice::doUploadCompressed(CompressedTransfer const &transfer) {
    collectGarbage();
    if ((info().limits.compressedFormats & compressedBit(transfer.format)) == 0) {
        throw DeviceError{std::string{"uploadImage: "} + properties.deviceName + " does not sample "
                          + compressedFormatInfo(transfer.format).name + " textures"};
    }
    auto const extent3 = ivec3{transfer.extent, static_cast<std::int32_t>(transfer.layers)};
    auto &mirror = images[transfer.identity->key()];
    auto const isNew = mirror.image.get() == VK_NULL_HANDLE or mirror.compressed != transfer.format
                    or mirror.dim != transfer.dim or not(mirror.extent == extent3) or mirror.levels != transfer.levels;
    if (isNew) {
        allocateImage(mirror, *transfer.identity, ImageFormat{}, transfer.format, transfer.dim, extent3, transfer.levels);
    }
    auto total = VkDeviceSize{0};
    for (std::uint32_t level = 0; level < transfer.levels; ++level) {
        total += (transfer.image->blocks(level).size() + 15) / 16 * 16;
    }
    auto const slice = stagingSlice(total);
    auto regions = std::vector<VkBufferImageCopy>{};
    auto offset = VkDeviceSize{0};
    for (std::uint32_t level = 0; level < transfer.levels; ++level) {
        auto const &blocks = transfer.image->blocks(level);
        std::memcpy(slice.mapped + offset, blocks.data(), blocks.size());
        auto const size = transfer.image->extent(level);
        auto region = VkBufferImageCopy{};
        region.bufferOffset = slice.offset + offset;
        region.imageSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, level, 0, transfer.layers};
        region.imageExtent = VkExtent3D{static_cast<std::uint32_t>(size.x), static_cast<std::uint32_t>(size.y), 1};
        regions.push_back(region);
        offset += (blocks.size() + 15) / 16 * 16;
    }
    auto const cmd = commandFor({detail::ResourceAccess{transfer.identity->key(), 0, 0, false, true}});
    vkCmdCopyBufferToImage(cmd, slice.buffer, mirror.image.get(), VK_IMAGE_LAYOUT_GENERAL,
                           static_cast<std::uint32_t>(regions.size()), regions.data());
}

void VulkanDevice::doDownloadImage(ImageTransfer const &transfer) {
    auto const it = images.find(transfer.identity->key());
    if (it == images.end() or not(it->second.format == transfer.format) or not(it->second.extent == transfer.extent)
        or it->second.levels != transfer.levels) {
        throw DeviceError{"downloadImage: the image was never uploaded to this device"};
    }
    auto offsets = std::vector<VkDeviceSize>{};
    auto total = VkDeviceSize{0};
    auto const last = transfer.firstLevel + transfer.levelCount;
    for (auto level = transfer.firstLevel; level < last; ++level) {
        offsets.push_back(total);
        total += (levelBytes(transfer, level) + 15) / 16 * 16;
    }
    auto const slice = stagingSlice(total);
    auto regions = std::vector<VkBufferImageCopy>{};
    for (auto level = transfer.firstLevel; level < last; ++level) {
        regions.push_back(imageCopy(transfer, level, slice.offset + offsets[level - transfer.firstLevel]));
    }
    auto const cmd = commandFor({detail::ResourceAccess{transfer.identity->key(), 0, 0, true, false}});
    vkCmdCopyImageToBuffer(cmd, it->second.image.get(), VK_IMAGE_LAYOUT_GENERAL, slice.buffer,
                           static_cast<std::uint32_t>(regions.size()), regions.data());
    hostReadBarrier(cmd);
    flush(); // the slice keeps its bytes until the next batch is recorded
    for (auto level = transfer.firstLevel; level < last; ++level) {
        auto const offset = offsets[level - transfer.firstLevel];
        transfer.readTexels(level, ConstByteSpan{slice.mapped + offset, levelBytes(transfer, level)});
    }
}

// A view of levels [first, first + count) and every layer.
DeviceObject<VkImageView> VulkanDevice::createView(ImageMirror const &mirror, std::uint32_t const first,
                                                   std::uint32_t const count) const {
    auto viewInfo = vkStruct<VkImageViewCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
    viewInfo.image = mirror.image.get();
    viewInfo.viewType = mirror.dim == Dim::D3        ? VK_IMAGE_VIEW_TYPE_3D
                      : mirror.dim == Dim::D2Array ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                                                   : VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = mirror.compressed ? vkCompressedFormat(*mirror.compressed) : vkFormatOf(mirror.format);
    viewInfo.subresourceRange =
        VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, first, count, 0, layerCount(mirror.dim, mirror.extent)};
    auto view = VkImageView{};
    check(vkCreateImageView(device.handle, &viewInfo, nullptr, &view), "vkCreateImageView");
    return DeviceObject<VkImageView>{device.handle, view, vkDestroyImageView};
}

// What storage image bindings of mip level `level` use.
VkImageView VulkanDevice::levelView(ImageMirror &mirror, std::uint32_t const level) {
    if (mirror.levels == 1) {
        return mirror.view.get();
    }
    auto &view = mirror.levelViews[level];
    if (view.get() == VK_NULL_HANDLE) {
        view = createView(mirror, level, 1);
    }
    return view.get();
}

KernelProgram const &VulkanDevice::programFor(std::string_view const name) {
    auto const key = std::string{name};
    auto const reflection = ProgramRegistry::instance().find(name);
    if (auto const it = programs.find(key); it != programs.end()) {
        if (not reflection or it->second.revision == reflection->revision) {
            return it->second;
        }
        // Reloaded from disk: queued work may still run the old pipelines.
        for (auto pipeline = pipelines.begin(); pipeline != pipelines.end();) {
            if (pipeline->first == key or pipeline->first.rfind(key + "|", 0) == 0) {
                bury(std::move(pipeline->second));
                pipeline = pipelines.erase(pipeline);
            } else {
                ++pipeline;
            }
        }
        programs.erase(it);
    }
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
    if (auto const reason = exceedsLimits(program, info().limits)) {
        throw DeviceError{*reason};
    }
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
    boundExternalImages.clear();
    uniformsSet.clear();
}

void VulkanDevice::doBindBuffer(BufferBindRequest const &request) {
    auto const it = buffers.find(request.identity->key());
    if (it == buffers.end()) {
        throw DeviceError{"bindBuffer: binding " + std::to_string(request.binding)
                          + " refers to a buffer that was never uploaded"};
    }
    if (request.range) {
        auto const alignment = properties.limits.minStorageBufferOffsetAlignment;
        if (request.range->offset % alignment != 0) {
            throw DeviceError{"bindBuffer: binding " + std::to_string(request.binding) + " starts at byte "
                              + std::to_string(request.range->offset) + ", not a multiple of " + std::to_string(alignment)
                              + " (DeviceLimits::storageBufferOffsetAlignment)"};
        }
        if (request.range->offset + request.range->size > it->second.size) {
            throw DeviceError{"bindBuffer: binding " + std::to_string(request.binding)
                              + " extends past the device buffer"};
        }
    }
    boundBuffers[request.binding] = BoundBuffer{request.identity->key(), request.range};
}

void VulkanDevice::doBindImage(ImageBindRequest const &request) {
    auto const it = images.find(request.identity->key());
    if (it == images.end() or not(it->second.format == request.format) or request.level >= it->second.levels) {
        throw DeviceError{"bindImage: binding " + std::to_string(request.binding)
                          + " refers to an image that was never uploaded"};
    }
    if ((it->second.usage & VK_IMAGE_USAGE_STORAGE_BIT) == 0) {
        throw DeviceError{"bindImage: format " + glslFormatQualifier(request.format)
                          + " cannot be a storage image on " + properties.deviceName + "; read it as a texture"};
    }
    boundImages[request.binding] = BoundImage{request.identity->key(), request.level};
}

void VulkanDevice::doBindTexture(TextureBindRequest const &request) {
    auto const it = images.find(request.identity->key());
    auto const matches = it != images.end() and it->second.dim == request.dim
                     and (request.compressed ? it->second.compressed == request.compressed
                                             : not it->second.compressed and it->second.format == request.format);
    if (not matches) {
        throw DeviceError{"bindTexture: binding " + std::to_string(request.binding)
                          + " refers to a texture that was never uploaded"};
    }
    if ((it->second.usage & VK_IMAGE_USAGE_SAMPLED_BIT) == 0) {
        throw DeviceError{"bindTexture: format " + glslFormatQualifier(request.format) + " cannot be sampled on "
                          + properties.deviceName};
    }
    auto const vkFormat = request.compressed ? vkCompressedFormat(*request.compressed) : vkFormatOf(request.format);
    auto const linearOk = (formatFeatures(vkFormat) & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
    if ((request.filter == Filter::Linear or request.mip == Mip::Linear) and not linearOk) {
        throw DeviceError{"bindTexture: format " + glslFormatQualifier(request.format)
                          + " cannot be filtered linearly on " + properties.deviceName};
    }
    boundTextures[request.binding] = BoundTexture{request.identity->key(), request.filter, request.wrap, request.mip};
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

void VulkanDevice::doBindExternalImage(ExternalImageBindRequest const &request) {
    if (not context->externalImages) {
        throw DeviceError{"bindExternalImage: " + std::string{properties.deviceName}
                          + " cannot import hardware buffers (needs Vulkan 1.1 and "
                            "VK_ANDROID_external_memory_android_hardware_buffer)"};
    }
    collectGarbage();
    auto const &identity = request.image->identity();
    auto &mirror = externalImages[identity.key()];
    if (mirror.image.get() == VK_NULL_HANDLE) {
        mirror.owner = identity.watch();
        importStorageFrame(mirror, *request.image, request.format);
    }
    boundExternalImages[request.binding] = identity.key();
}

// Imports an RGBA AHardwareBuffer as a storage image, if the driver says it
// can write that buffer's memory that way.
void VulkanDevice::importStorageFrame(ExternalImageMirror &mirror, ExternalImage const &image,
                                      ImageFormat const &format) {
    auto *const buffer = image.hardwareBuffer();
    auto const vkFormat = vkFormatOf(format);
    auto const usage = VkImageUsageFlags{VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT};
    auto externalInfo =
        vkStruct<VkPhysicalDeviceExternalImageFormatInfo>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO);
    externalInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
    auto formatInfo = vkStruct<VkPhysicalDeviceImageFormatInfo2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2);
    formatInfo.pNext = &externalInfo;
    formatInfo.format = vkFormat;
    formatInfo.type = VK_IMAGE_TYPE_2D;
    formatInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    formatInfo.usage = usage;
    auto externalProperties = vkStruct<VkExternalImageFormatProperties>(VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES);
    auto formatProperties = vkStruct<VkImageFormatProperties2>(VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2);
    formatProperties.pNext = &externalProperties;
    auto const supported = vkGetPhysicalDeviceImageFormatProperties2(physical, &formatInfo, &formatProperties);
    if (supported != VK_SUCCESS
        or (externalProperties.externalMemoryProperties.externalMemoryFeatures
            & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT)
               == 0) {
        throw DeviceError{"bindExternalImage: " + std::string{properties.deviceName} + " cannot write "
                          + glslFormatQualifier(format) + " hardware buffers as storage images"};
    }

    auto const getProperties = reinterpret_cast<PFN_vkGetAndroidHardwareBufferPropertiesANDROID>(
        vkGetDeviceProcAddr(device.handle, "vkGetAndroidHardwareBufferPropertiesANDROID"));
    auto bufferFormat = vkStruct<VkAndroidHardwareBufferFormatPropertiesANDROID>(
        VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID);
    auto bufferProperties =
        vkStruct<VkAndroidHardwareBufferPropertiesANDROID>(VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID);
    bufferProperties.pNext = &bufferFormat;
    check(getProperties(device.handle, buffer, &bufferProperties), "vkGetAndroidHardwareBufferPropertiesANDROID");
    if (bufferFormat.format != vkFormat) {
        throw DeviceError{"bindExternalImage: the hardware buffer's Vulkan format does not match "
                          + glslFormatQualifier(format)};
    }

    auto desc = AHardwareBuffer_Desc{};
    AHardwareBuffer_describe(buffer, &desc);
    auto externalMemory = vkStruct<VkExternalMemoryImageCreateInfo>(VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO);
    externalMemory.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
    auto imageInfo = vkStruct<VkImageCreateInfo>(VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);
    imageInfo.pNext = &externalMemory;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = vkFormat;
    imageInfo.extent = VkExtent3D{desc.width, desc.height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = usage;
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
    viewInfo.image = rawImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = vkFormat;
    viewInfo.subresourceRange = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    auto view = VkImageView{};
    check(vkCreateImageView(device.handle, &viewInfo, nullptr, &view), "vkCreateImageView (hardware buffer)");
    mirror.view = DeviceObject<VkImageView>{device.handle, view, vkDestroyImageView};
}
#endif

VkFormatFeatureFlags VulkanDevice::formatFeatures(VkFormat const format) const {
    auto formatProperties = VkFormatProperties{};
    vkGetPhysicalDeviceFormatProperties(physical, format, &formatProperties);
    return formatProperties.optimalTilingFeatures;
}

VkSampler VulkanDevice::samplerFor(Filter const filter, Wrap const wrap, Mip const mip) {
    auto const key = std::make_tuple(filter, wrap, mip);
    auto it = samplers.find(key);
    if (it == samplers.end()) {
        auto info = vkStruct<VkSamplerCreateInfo>(VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO);
        info.magFilter = filter == Filter::Linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        info.minFilter = info.magFilter;
        info.mipmapMode = mip == Mip::Linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
        info.addressModeU = addressMode(wrap);
        info.addressModeV = addressMode(wrap);
        info.addressModeW = addressMode(wrap);
        info.maxLod = mip == Mip::None ? 0.0f : VK_LOD_CLAMP_NONE;
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
    for (int axis = 0; axis < 3 and not dispatch.indirect; ++axis) {
        if (dispatch.workGroups[axis] > limits.maxComputeWorkGroupCount[axis]) {
            throw DeviceError{"workgroup count exceeds the device limit"};
        }
    }
    auto indirectBuffer = VkBuffer{VK_NULL_HANDLE};
    if (dispatch.indirect) {
        auto const it = buffers.find(dispatch.indirect->identity->key());
        if (it == buffers.end() or it->second.size < dispatch.indirect->offset + sizeof(DispatchIndirectCommand)) {
            throw DeviceError{"executeIndirect: the argument buffer was never uploaded to this device"};
        }
        indirectBuffer = it->second.handle();
    }
    if (timersEnabled() and recordingBatch().queriesUsed + 2 > kQueryCapacity) {
        submitBatch(); // before allocating the descriptor set, which belongs to the batch
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
    auto accesses = std::vector<detail::ResourceAccess>{};
    auto used = std::vector<void const *>{};
    auto const access = [&](void const *const key, std::optional<ByteRange> const &range, Access const how) {
        accesses.push_back(this->access(key, range ? range->offset : 0, range ? range->size : 0,
                                        how != Access::WriteOnly, how != Access::ReadOnly));
        used.push_back(key);
    };
    if (dispatch.indirect) {
        access(dispatch.indirect->identity->key(), ByteRange{dispatch.indirect->offset, sizeof(DispatchIndirectCommand)},
               Access::ReadOnly);
    }
    struct StorageFrame {
        VkImage image = VK_NULL_HANDLE;
        bool wasAcquired = false;
    };
    auto storageFrames = std::vector<StorageFrame>{};
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
            auto const &range = bound->second.range;
            bufferInfos.push_back(VkDescriptorBufferInfo{buffers.at(bound->second.key).handle(),
                                                         range ? range->offset : 0,
                                                         range ? range->size : VK_WHOLE_SIZE});
            access(bound->second.key, range, slot.access);
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &bufferInfos.back();
        } else if (slot.kind == ResourceKind::SampledTexture) {
            auto const bound = boundTextures.find(slot.binding);
            if (bound == boundTextures.end()) {
                throw DeviceError{"execute: texture '" + name + " is not bound"};
            }
            auto const &texture = bound->second;
            imageInfos.push_back(VkDescriptorImageInfo{samplerFor(texture.filter, texture.wrap, texture.mip),
                                                       images.at(texture.key).view.get(), VK_IMAGE_LAYOUT_GENERAL});
            access(texture.key, std::nullopt, Access::ReadOnly);
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &imageInfos.back();
        } else if (slot.kind == ResourceKind::ExternalImage) {
            auto const bound = boundExternalImages.find(slot.binding);
            if (bound == boundExternalImages.end()) {
                throw DeviceError{"execute: external image '" + name + " is not bound"};
            }
            auto &frame = externalImages.at(bound->second);
            imageInfos.push_back(VkDescriptorImageInfo{VK_NULL_HANDLE, frame.view.get(), VK_IMAGE_LAYOUT_GENERAL});
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write.pImageInfo = &imageInfos.back();
            storageFrames.push_back({frame.image.get(), frame.acquired});
            frame.acquired = true;
            access(bound->second, std::nullopt, slot.access);
        } else if (slot.kind == ResourceKind::ExternalTexture) {
            auto const &frame = externals.at(boundExternals.at(slot.binding));
            imageInfos.push_back(VkDescriptorImageInfo{frame.sampler.get(), frame.view.get(),
                                                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
            access(boundExternals.at(slot.binding), std::nullopt, Access::ReadOnly);
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.pImageInfo = &imageInfos.back();
        } else {
            auto const bound = boundImages.find(slot.binding);
            if (bound == boundImages.end()) {
                throw DeviceError{"execute: image '" + name + " is not bound"};
            }
            imageInfos.push_back(VkDescriptorImageInfo{VK_NULL_HANDLE, levelView(images.at(bound->second.key), bound->second.level),
                                                       VK_IMAGE_LAYOUT_GENERAL});
            access(bound->second.key, std::nullopt, slot.access);
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            write.pImageInfo = &imageInfos.back();
        }
        writes.push_back(write);
    }
    vkUpdateDescriptorSets(device.handle, static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);

    auto const layout = pipeline.layout.get();
    auto const cmd = recording();
    if (timersEnabled()) {
        // Timed dispatches run one at a time, so each pair of timestamps
        // brackets only its own dispatch.
        fullBarrier(cmd);
        hazards.barrierRecorded();
    }
    if (hazards.needsBarrier(accesses)) {
        fullBarrier(cmd);
    }
    for (auto const *key : used) {
        claim(cmd, key);
    }
    for (auto const image : frames) {
        acquireForeignFrame(cmd, image);
    }
    for (auto const &frame : storageFrames) {
        transferForeignImage(cmd, frame.image, queueFamily, true,
                             frame.wasAcquired ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline.get());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
    if (reflection.pushConstantSize > 0) {
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, reflection.pushConstantSize,
                           pushConstants.data());
    }
    auto const record = [&] {
        if (dispatch.indirect) {
            vkCmdDispatchIndirect(cmd, indirectBuffer, dispatch.indirect->offset);
        } else {
            vkCmdDispatch(cmd, dispatch.workGroups.x, dispatch.workGroups.y, dispatch.workGroups.z);
        }
    };
    auto const releaseFrames = [&] {
        for (auto const &frame : storageFrames) {
            transferForeignImage(cmd, frame.image, queueFamily, false);
        }
    };
    if (not timersEnabled()) {
        record();
        releaseFrames();
        return;
    }
    // Bottom-of-pipe for both: each is written once every earlier command
    // has completed, so the pair brackets exactly this dispatch.
    auto &batch = recordingBatch();
    auto const first = reserveTimestamps(cmd);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, batch.queryPool.get(), first);
    record();
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, batch.queryPool.get(), first + 1);
    releaseFrames();
    batch.pendingTimings.push_back(PendingTiming{std::string{dispatch.programName},
                                                 dispatch.indirect ? uvec3{} : dispatch.workGroups, first,
                                                 dispatch.indirect.has_value()});
}

} // namespace

std::unique_ptr<Device> createDevice(VulkanDeviceConfig const &config) {
    return std::make_unique<VulkanDevice>(config);
}

} // namespace tc::vulkan
