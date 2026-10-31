// Buffers in memory another device exported: a plain Vulkan device of the
// same GPU allocates host-visible memory, tc imports it, and both see the
// same words.

#include <array>
#include <cstring>
#include <memory>
#include <optional>
#include <vector>

#include <vulkan/vulkan.h>

#include "equivalence.hpp"
#include "kernels/memory_kernels.hpp"

namespace {

using namespace tc;
using namespace tc::testing_support;

constexpr int kWords = 256;
// llvmpipe's OpenGL rejects a buffer that fills the whole imported
// allocation (GL_OUT_OF_MEMORY); 64 more bytes are enough.
constexpr VkDeviceSize kSlack = 64;

// A Vulkan device of one GPU with an exportable, mapped allocation.
struct Exporter {
    static std::unique_ptr<Exporter> forGpu(GpuIdentity const &gpu) {
        auto exporter = std::make_unique<Exporter>();
        auto app = VkApplicationInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = VK_API_VERSION_1_1;
        auto instanceInfo = VkInstanceCreateInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        instanceInfo.pApplicationInfo = &app;
        if (vkCreateInstance(&instanceInfo, nullptr, &exporter->instance) != VK_SUCCESS) {
            return nullptr;
        }
        auto count = std::uint32_t{0};
        vkEnumeratePhysicalDevices(exporter->instance, &count, nullptr);
        auto devices = std::vector<VkPhysicalDevice>(count);
        vkEnumeratePhysicalDevices(exporter->instance, &count, devices.data());
        for (auto const candidate : devices) {
            auto ids = VkPhysicalDeviceIDProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            auto properties = VkPhysicalDeviceProperties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            properties.pNext = &ids;
            vkGetPhysicalDeviceProperties2(candidate, &properties);
            if (std::memcmp(ids.deviceUUID, gpu.device.data(), VK_UUID_SIZE) == 0
                and std::memcmp(ids.driverUUID, gpu.driver.data(), VK_UUID_SIZE) == 0) {
                exporter->physical = candidate;
            }
        }
        if (exporter->physical == VK_NULL_HANDLE) {
            return nullptr;
        }
        auto const priority = 1.0f;
        auto queue = VkDeviceQueueCreateInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queue.queueCount = 1;
        queue.pQueuePriorities = &priority;
        char const *extensions[] = {VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
        auto deviceInfo = VkDeviceCreateInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queue;
        deviceInfo.enabledExtensionCount = 1;
        deviceInfo.ppEnabledExtensionNames = extensions;
        if (vkCreateDevice(exporter->physical, &deviceInfo, nullptr, &exporter->device) != VK_SUCCESS) {
            return nullptr;
        }
        return exporter;
    }

    // `bytes` of host-visible memory, mapped at `words`, exported as an opaque fd.
    std::optional<ExternalBufferMemory> exportBuffer(std::size_t const bytes) {
        auto externalInfo = VkExternalMemoryBufferCreateInfo{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        externalInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        auto bufferInfo = VkBufferCreateInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufferInfo.pNext = &externalInfo;
        bufferInfo.size = bytes;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            return std::nullopt;
        }
        auto requirements = VkMemoryRequirements{};
        vkGetBufferMemoryRequirements(device, buffer, &requirements);
        auto memoryProperties = VkPhysicalDeviceMemoryProperties{};
        vkGetPhysicalDeviceMemoryProperties(physical, &memoryProperties);
        auto constexpr wanted = VkMemoryPropertyFlags{VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
        auto type = memoryProperties.memoryTypeCount;
        for (std::uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
            if ((requirements.memoryTypeBits & (1u << i)) != 0
                and (memoryProperties.memoryTypes[i].propertyFlags & wanted) == wanted) {
                type = i;
                break;
            }
        }
        if (type == memoryProperties.memoryTypeCount) {
            return std::nullopt;
        }
        auto exportInfo = VkExportMemoryAllocateInfo{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
        exportInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        auto allocateInfo = VkMemoryAllocateInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocateInfo.pNext = &exportInfo;
        allocateInfo.allocationSize = requirements.size + kSlack;
        allocateInfo.memoryTypeIndex = type;
        if (vkAllocateMemory(device, &allocateInfo, nullptr, &memory) != VK_SUCCESS
            or vkBindBufferMemory(device, buffer, memory, 0) != VK_SUCCESS
            or vkMapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
            return std::nullopt;
        }
        auto const getFd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR"));
        auto fdInfo = VkMemoryGetFdInfoKHR{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
        fdInfo.memory = memory;
        fdInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
        auto fd = -1;
        if (getFd == nullptr or getFd(device, &fdInfo, &fd) != VK_SUCCESS) {
            return std::nullopt;
        }
        auto result = ExternalBufferMemory{};
        result.fd = MemoryFd{fd};
        result.allocationBytes = requirements.size + kSlack;
        result.memoryType = type;
        return result;
    }

    uint *words() const noexcept { return static_cast<uint *>(mapped); }

    Exporter() = default;
    Exporter(Exporter const &) = delete;
    Exporter &operator=(Exporter const &) = delete;
    ~Exporter() {
        if (device != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, buffer, nullptr);
            vkFreeMemory(device, memory, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance != VK_NULL_HANDLE) {
            vkDestroyInstance(instance, nullptr);
        }
    }

    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void *mapped = nullptr;
};

TEST_P(Equivalence, ImportedBuffersShareMemory) {
    auto const info = gpu->info();
    if (not info.importsMemory or not info.gpu) {
        GTEST_SKIP() << info.deviceName << " (" << info.backend << ") cannot import memory";
    }
    auto exporter = Exporter::forGpu(*info.gpu);
    if (exporter == nullptr) {
        GTEST_SKIP() << "no Vulkan device has the UUIDs of " << info.deviceName;
    }
    auto memory = exporter->exportBuffer(2 * kWords * sizeof(uint));
    if (not memory) {
        GTEST_SKIP() << "the Vulkan device of " << info.deviceName << " cannot export host-visible memory";
    }
    for (int i = 0; i < 2 * kWords; ++i) {
        exporter->words()[i] = static_cast<uint>(i * 3 + 1);
    }
    EXPECT_THROW(cpu->importBuffer<uint>(ExternalBufferMemory{}, kWords), DeviceError);
    auto imported = gpu->importBuffer<uint>(std::move(*memory), 2 * kWords);
    EXPECT_FALSE(imported.hostView()->hasHostData());

    auto out = gpu->createBuffer<uint>(kWords);
    auto sink = BufferResource<uint>{kWords};
    gpu->uploadBuffer(out);
    gpu->uploadBuffer(sink);
    auto copy = tests::SlowCopy{};
    copy.in.attach(imported, 0, kWords);
    copy.out.attach(out);
    copy.sink.attach(sink);
    copy.addend = 0u;
    copy.rounds = 4u;
    gpu->useKernel(copy);
    gpu->bindBuffer(copy.in);
    gpu->bindBuffer(copy.out);
    gpu->bindBuffer(copy.sink);
    gpu->setUniform(copy.addend);
    gpu->setUniform(copy.rounds);
    gpu->execute(copy, uvec3{kWords / 64u, 1u, 1u});
    auto fill = tests::FillWords{};
    fill.out.attach(imported, kWords, kWords);
    fill.seed = 9u;
    gpu->useKernel(fill);
    gpu->bindBuffer(fill.out);
    gpu->setUniform(fill.seed);
    gpu->execute(fill, uvec3{kWords / 64u, 1u, 1u});
    gpu->finish();
    for (int i = 0; i < kWords; ++i) {
        ASSERT_EQ(out[i], static_cast<uint>(i * 3 + 1)) << "tc read the exporter's word " << i;
        ASSERT_EQ(exporter->words()[kWords + i], static_cast<uint>(i * 9 + 1)) << "the exporter sees tc's word " << i;
    }
}

TEST_P(Equivalence, DevicesReportTheirGpu) {
    EXPECT_FALSE(cpu->info().gpu);
    EXPECT_FALSE(cpu->info().importsMemory);
    auto const info = gpu->info();
    if (info.backend == "vulkan") {
        ASSERT_TRUE(info.gpu) << "Vulkan 1.1 has device and driver UUIDs";
        EXPECT_NE(info.gpu->device, (std::array<std::uint8_t, 16>{}));
    }
}

} // namespace
