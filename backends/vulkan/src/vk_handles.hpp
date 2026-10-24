#pragma once

#include <vulkan/vulkan.h>

#include <string>
#include <utility>

#include <tc/device.hpp>

namespace tc::vulkan {

// Throws DeviceError naming the failing call when `result` is an error.
inline void check(VkResult const result, char const *operation) {
    if (result != VK_SUCCESS) {
        throw DeviceError{std::string{operation} + " failed with VkResult " + std::to_string(result)};
    }
}

// A zero-initialised Vulkan create-info / barrier struct with its sType set.
template <class T>
T vkStruct(VkStructureType const type) noexcept {
    auto value = T{};
    value.sType = type;
    return value;
}

// Move-only owner of a handle created from a VkDevice.
template <class H>
struct DeviceObject {
    using Destroy = void(VKAPI_PTR *)(VkDevice, H, VkAllocationCallbacks const *);

    DeviceObject() = default;
    DeviceObject(VkDevice const device_, H const handle_, Destroy const destroy_) noexcept
        : device(device_), handle(handle_), destroy(destroy_) {}
    DeviceObject(DeviceObject &&other) noexcept
        : device(other.device), handle(std::exchange(other.handle, VK_NULL_HANDLE)), destroy(other.destroy) {}
    DeviceObject &operator=(DeviceObject &&other) noexcept {
        std::swap(device, other.device);
        std::swap(handle, other.handle);
        std::swap(destroy, other.destroy);
        return *this;
    }
    DeviceObject(DeviceObject const &) = delete;
    DeviceObject &operator=(DeviceObject const &) = delete;
    ~DeviceObject() {
        if (handle != VK_NULL_HANDLE) {
            destroy(device, handle, nullptr);
        }
    }

    H get() const noexcept { return handle; }

private:
    VkDevice device = VK_NULL_HANDLE;
    H handle = VK_NULL_HANDLE;
    Destroy destroy = nullptr;
};

struct InstanceOwner {
    VkInstance handle = VK_NULL_HANDLE;

    InstanceOwner() = default;
    InstanceOwner(InstanceOwner const &) = delete;
    InstanceOwner &operator=(InstanceOwner const &) = delete;
    ~InstanceOwner() {
        if (handle != VK_NULL_HANDLE) {
            vkDestroyInstance(handle, nullptr);
        }
    }
};

struct DeviceOwner {
    VkDevice handle = VK_NULL_HANDLE;

    DeviceOwner() = default;
    DeviceOwner(DeviceOwner const &) = delete;
    DeviceOwner &operator=(DeviceOwner const &) = delete;
    ~DeviceOwner() {
        if (handle != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(handle);
            vkDestroyDevice(handle, nullptr);
        }
    }
};

} // namespace tc::vulkan
