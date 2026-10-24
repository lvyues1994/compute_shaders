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

struct BufferMirror {
    std::weak_ptr<void const> owner;
    DeviceObject<VkDeviceMemory> memory; // declared before the buffer: freed after it
    DeviceObject<VkBuffer> buffer;
    VkDeviceSize size = 0;
};

struct ImageMirror {
    std::weak_ptr<void const> owner;
    DeviceObject<VkDeviceMemory> memory;
    DeviceObject<VkImage> image;
    DeviceObject<VkImageView> view;
    ImageFormat format;
    Dim dim = Dim::D2;
    ivec3 extent{};
};

struct PipelineObject {
    KernelProgram reflection;
    DeviceObject<VkShaderModule> module;
    DeviceObject<VkDescriptorSetLayout> setLayout;
    DeviceObject<VkPipelineLayout> layout;
    DeviceObject<VkPipeline> pipeline;
};

struct VulkanDevice final : Device {
    explicit VulkanDevice(VulkanDeviceConfig const &config);

    DeviceInfo info() const override { return DeviceInfo{"vulkan", properties.deviceName}; }

protected:
    void doUploadBuffer(ResourceIdentity const &identity, ConstByteSpan bytes) override;
    void doDownloadBuffer(ResourceIdentity const &identity, ByteSpan bytes) override;
    void doUploadImage(ImageTransfer const &transfer) override;
    void doDownloadImage(ImageTransfer const &transfer) override;
    void doUseKernel(KernelUse const &use) override;
    void doBindBuffer(BufferBindRequest const &request) override;
    void doBindImage(ImageBindRequest const &request) override;
    void doSetUniform(UniformWrite const &write) override;
    void doExecute(Dispatch const &dispatch) override;

private:
    void createInstance(VulkanDeviceConfig const &config);
    void pickPhysicalDevice(VulkanDeviceConfig const &config);
    void createLogicalDevice();
    std::uint32_t memoryType(std::uint32_t typeBits, VkMemoryPropertyFlags required) const;
    DeviceObject<VkDeviceMemory> allocate(VkMemoryRequirements const &requirements, VkMemoryPropertyFlags flags);
    void ensureStaging(VkDeviceSize size);
    template <class Record>
    void submit(Record const &record);
    void collectGarbage();
    PipelineObject &pipelineFor(std::string_view name);

    InstanceOwner instance;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    std::uint32_t queueFamily = 0;
    DeviceOwner device; // destroyed after every DeviceObject declared below
    VkQueue queue = VK_NULL_HANDLE;
    DeviceObject<VkCommandPool> commandPool;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    DeviceObject<VkFence> fence;
    DeviceObject<VkDescriptorPool> descriptorPool;
    DeviceObject<VkDeviceMemory> stagingMemory;
    DeviceObject<VkBuffer> staging;
    std::byte *stagingMapped = nullptr;
    VkDeviceSize stagingCapacity = 0;
    std::unordered_map<void const *, BufferMirror> buffers;
    std::unordered_map<void const *, ImageMirror> images;
    std::unordered_map<std::string, PipelineObject> pipelines;

    PipelineObject *current = nullptr;
    std::map<std::uint32_t, void const *> boundBuffers;
    std::map<std::uint32_t, void const *> boundImages;
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

    VkDescriptorPoolSize const sizes[] = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64},
                                          {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64}};
    auto descriptorInfo = vkStruct<VkDescriptorPoolCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO);
    descriptorInfo.maxSets = 1;
    descriptorInfo.poolSizeCount = 2;
    descriptorInfo.pPoolSizes = sizes;
    auto rawPool = VkDescriptorPool{};
    check(vkCreateDescriptorPool(device.handle, &descriptorInfo, nullptr, &rawPool), "vkCreateDescriptorPool");
    descriptorPool = DeviceObject<VkDescriptorPool>{device.handle, rawPool, vkDestroyDescriptorPool};
}

void VulkanDevice::createInstance(VulkanDeviceConfig const &config) {
    auto app = vkStruct<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
    app.pApplicationName = "tc";
    app.pEngineName = "tc";
    app.apiVersion = VK_API_VERSION_1_0;
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

DeviceObject<VkDeviceMemory> VulkanDevice::allocate(VkMemoryRequirements const &requirements,
                                                    VkMemoryPropertyFlags const flags) {
    auto info = vkStruct<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    info.allocationSize = requirements.size;
    info.memoryTypeIndex = memoryType(requirements.memoryTypeBits, flags);
    auto memory = VkDeviceMemory{};
    check(vkAllocateMemory(device.handle, &info, nullptr, &memory), "vkAllocateMemory");
    return DeviceObject<VkDeviceMemory>{device.handle, memory, vkFreeMemory};
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

template <class Record>
void VulkanDevice::submit(Record const &record) {
    check(vkResetCommandBuffer(commandBuffer, 0), "vkResetCommandBuffer");
    auto begin = vkStruct<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(commandBuffer, &begin), "vkBeginCommandBuffer");
    fullBarrier(commandBuffer);
    record(commandBuffer);
    check(vkEndCommandBuffer(commandBuffer), "vkEndCommandBuffer");
    auto info = vkStruct<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
    info.commandBufferCount = 1;
    info.pCommandBuffers = &commandBuffer;
    auto const rawFence = fence.get();
    check(vkQueueSubmit(queue, 1, &info, rawFence), "vkQueueSubmit");
    check(vkWaitForFences(device.handle, 1, &rawFence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
    check(vkResetFences(device.handle, 1, &rawFence), "vkResetFences");
}

void VulkanDevice::collectGarbage() {
    for (auto it = buffers.begin(); it != buffers.end();) {
        it = it->second.owner.expired() ? buffers.erase(it) : std::next(it);
    }
    for (auto it = images.begin(); it != images.end();) {
        it = it->second.owner.expired() ? images.erase(it) : std::next(it);
    }
}

void VulkanDevice::doUploadBuffer(ResourceIdentity const &identity, ConstByteSpan const bytes) {
    if (bytes.empty()) {
        throw DeviceError{"uploadBuffer: Vulkan buffers cannot be empty"};
    }
    collectGarbage();
    auto &mirror = buffers[identity.key()];
    if (mirror.size != bytes.size()) {
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
    ensureStaging(bytes.size());
    std::memcpy(stagingMapped, bytes.data(), bytes.size());
    auto const target = mirror.buffer.get();
    auto const source = staging.get();
    submit([&](VkCommandBuffer const cmd) {
        auto const region = VkBufferCopy{0, 0, bytes.size()};
        vkCmdCopyBuffer(cmd, source, target, 1, &region);
    });
}

void VulkanDevice::doDownloadBuffer(ResourceIdentity const &identity, ByteSpan const bytes) {
    auto const it = buffers.find(identity.key());
    if (it == buffers.end() or it->second.size != bytes.size()) {
        throw DeviceError{"downloadBuffer: the buffer was never uploaded to this device (or changed size)"};
    }
    ensureStaging(bytes.size());
    auto const source = it->second.buffer.get();
    auto const target = staging.get();
    submit([&](VkCommandBuffer const cmd) {
        auto const region = VkBufferCopy{0, 0, bytes.size()};
        vkCmdCopyBuffer(cmd, source, target, 1, &region);
        hostReadBarrier(cmd);
    });
    std::memcpy(bytes.data(), stagingMapped, bytes.size());
}

void VulkanDevice::doUploadImage(ImageTransfer const &transfer) {
    collectGarbage();
    auto const format = vkFormatOf(transfer.format);
    auto &mirror = images[transfer.identity->key()];
    auto const isNew = mirror.image.get() == VK_NULL_HANDLE or not(mirror.format == transfer.format)
                    or not(mirror.extent == transfer.extent);
    if (isNew) {
        auto formatProperties = VkFormatProperties{};
        vkGetPhysicalDeviceFormatProperties(physical, format, &formatProperties);
        if ((formatProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) == 0) {
            throw DeviceError{"image format " + glslFormatQualifier(transfer.format)
                              + " cannot be a storage image on " + properties.deviceName};
        }
        mirror = ImageMirror{};
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
        info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
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
    ensureStaging(size);
    transfer.writeTexels(ByteSpan{stagingMapped, size});
    auto const image = mirror.image.get();
    auto const source = staging.get();
    auto const extent = VkExtent3D{static_cast<std::uint32_t>(transfer.extent.x),
                                   static_cast<std::uint32_t>(transfer.extent.y),
                                   static_cast<std::uint32_t>(transfer.extent.z)};
    submit([&](VkCommandBuffer const cmd) {
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
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr,
                                 0, nullptr, 1, &barrier);
        }
        auto region = VkBufferImageCopy{};
        region.imageSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = extent;
        vkCmdCopyBufferToImage(cmd, source, image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    });
}

void VulkanDevice::doDownloadImage(ImageTransfer const &transfer) {
    auto const it = images.find(transfer.identity->key());
    if (it == images.end() or not(it->second.format == transfer.format) or not(it->second.extent == transfer.extent)) {
        throw DeviceError{"downloadImage: the image was never uploaded to this device"};
    }
    auto const texelCount = static_cast<std::size_t>(transfer.extent.x) * transfer.extent.y * transfer.extent.z;
    auto const size = texelCount * transfer.format.bytesPerTexel();
    ensureStaging(size);
    auto const image = it->second.image.get();
    auto const target = staging.get();
    auto const extent = VkExtent3D{static_cast<std::uint32_t>(transfer.extent.x),
                                   static_cast<std::uint32_t>(transfer.extent.y),
                                   static_cast<std::uint32_t>(transfer.extent.z)};
    submit([&](VkCommandBuffer const cmd) {
        auto region = VkBufferImageCopy{};
        region.imageSubresource = VkImageSubresourceLayers{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = extent;
        vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_GENERAL, target, 1, &region);
        hostReadBarrier(cmd);
    });
    transfer.readTexels(ConstByteSpan{stagingMapped, size});
}

PipelineObject &VulkanDevice::pipelineFor(std::string_view const name) {
    auto const key = std::string{name};
    if (auto const it = pipelines.find(key); it != pipelines.end()) {
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
    auto pipeline = PipelineObject{};
    pipeline.reflection = *reflection;

    auto moduleInfo = vkStruct<VkShaderModuleCreateInfo>(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
    moduleInfo.codeSize = reflection->spirv.size() * sizeof(std::uint32_t);
    moduleInfo.pCode = reflection->spirv.data();
    auto module = VkShaderModule{};
    check(vkCreateShaderModule(device.handle, &moduleInfo, nullptr, &module), "vkCreateShaderModule");
    pipeline.module = DeviceObject<VkShaderModule>{device.handle, module, vkDestroyShaderModule};

    auto bindings = std::vector<VkDescriptorSetLayoutBinding>{};
    for (auto const &slot : reflection->resources) {
        auto binding = VkDescriptorSetLayoutBinding{};
        binding.binding = slot.binding;
        binding.descriptorType = slot.kind == ResourceKind::StorageBuffer ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                                                          : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(binding);
    }
    auto setInfo = vkStruct<VkDescriptorSetLayoutCreateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO);
    setInfo.bindingCount = static_cast<std::uint32_t>(bindings.size());
    setInfo.pBindings = bindings.data();
    auto setLayout = VkDescriptorSetLayout{};
    check(vkCreateDescriptorSetLayout(device.handle, &setInfo, nullptr, &setLayout), "vkCreateDescriptorSetLayout");
    pipeline.setLayout = DeviceObject<VkDescriptorSetLayout>{device.handle, setLayout, vkDestroyDescriptorSetLayout};

    auto const range = VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, reflection->pushConstantSize};
    auto layoutInfo = vkStruct<VkPipelineLayoutCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &setLayout;
    layoutInfo.pushConstantRangeCount = reflection->pushConstantSize > 0 ? 1 : 0;
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
    auto &pipeline = pipelineFor(use.programName);
    auto const &size = pipeline.reflection.localSize;
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
    current = &pipeline;
    boundBuffers.clear();
    boundImages.clear();
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
    boundImages[request.binding] = request.identity->key();
}

void VulkanDevice::doSetUniform(UniformWrite const &write) {
    if (current == nullptr) {
        throw DeviceError{"setUniform: call useKernel() first"};
    }
    for (auto const &slot : current->reflection.uniforms) {
        if (slot.location == write.location) {
            std::memcpy(pushConstants.data() + slot.pushConstantOffset, write.bytes.data(), write.bytes.size());
            uniformsSet[slot.location] = true;
            return;
        }
    }
    throw DeviceError{"setUniform: kernel '" + std::string{current->reflection.name} + "' has no uniform at location "
                      + std::to_string(write.location)};
}

void VulkanDevice::doExecute(Dispatch const &dispatch) {
    if (current == nullptr or current->reflection.name != dispatch.programName) {
        throw DeviceError{"execute: call useKernel() for '" + std::string{dispatch.programName} + "' first"};
    }
    auto const &reflection = current->reflection;
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
    check(vkResetDescriptorPool(device.handle, descriptorPool.get(), 0), "vkResetDescriptorPool");
    auto set = VkDescriptorSet{};
    auto const setLayout = current->setLayout.get();
    auto allocInfo = vkStruct<VkDescriptorSetAllocateInfo>(VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO);
    allocInfo.descriptorPool = descriptorPool.get();
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &setLayout;
    check(vkAllocateDescriptorSets(device.handle, &allocInfo, &set), "vkAllocateDescriptorSets");

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
            bufferInfos.push_back(VkDescriptorBufferInfo{buffers.at(bound->second).buffer.get(), 0, VK_WHOLE_SIZE});
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &bufferInfos.back();
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

    auto const pipeline = current->pipeline.get();
    auto const layout = current->layout.get();
    auto const pushSize = reflection.pushConstantSize;
    submit([&](VkCommandBuffer const cmd) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, nullptr);
        if (pushSize > 0) {
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushConstants.data());
        }
        vkCmdDispatch(cmd, dispatch.workGroups.x, dispatch.workGroups.y, dispatch.workGroups.z);
    });
}

} // namespace

std::unique_ptr<Device> createDevice(VulkanDeviceConfig const &config) {
    return std::make_unique<VulkanDevice>(config);
}

} // namespace tc::vulkan
