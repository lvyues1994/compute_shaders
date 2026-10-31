#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <unistd.h>

namespace tc {

// Owns a file descriptor of exported device memory; empty when -1.
struct MemoryFd {
    MemoryFd() noexcept = default;
    explicit MemoryFd(int const fd_) noexcept : fd(fd_) {}
    MemoryFd(MemoryFd &&other) noexcept : fd(std::exchange(other.fd, -1)) {}
    MemoryFd &operator=(MemoryFd &&other) noexcept {
        if (this != &other) {
            reset();
            fd = std::exchange(other.fd, -1);
        }
        return *this;
    }
    MemoryFd(MemoryFd const &) = delete;
    MemoryFd &operator=(MemoryFd const &) = delete;
    ~MemoryFd() { reset(); }

    bool empty() const noexcept { return fd < 0; }
    int get() const noexcept { return fd; }
    // Gives up ownership, e.g. to the driver that imports it.
    int release() noexcept { return std::exchange(fd, -1); }

private:
    void reset() noexcept {
        if (fd >= 0) {
            ::close(fd);
        }
        fd = -1;
    }

    int fd = -1;
};

// Memory another device exported, for Device::importBuffer: an opaque file
// descriptor (VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT on Vulkan,
// GL_HANDLE_TYPE_OPAQUE_FD_EXT on OpenGL) of an allocation of
// `allocationBytes`, in which the buffer starts at `offset`. The exporting
// device must be the same GPU and driver (DeviceInfo::gpu). Importing
// hands the descriptor to the device.
struct ExternalBufferMemory {
    MemoryFd fd;
    std::size_t allocationBytes = 0;
    std::size_t offset = 0;
    // Vulkan: the memory type the exporter allocated from; both devices
    // enumerate the same types when they are the same GPU and driver.
    // Unused on OpenGL.
    std::uint32_t memoryType = 0;
};

// Identifies a GPU and its driver (Vulkan VkPhysicalDeviceIDProperties,
// OpenGL GL_EXT_memory_object): two devices share memory only when both
// UUIDs match.
struct GpuIdentity {
    std::array<std::uint8_t, 16> device{};
    std::array<std::uint8_t, 16> driver{};
};

inline bool operator==(GpuIdentity const &a, GpuIdentity const &b) noexcept {
    return a.device == b.device and a.driver == b.driver;
}

inline bool operator!=(GpuIdentity const &a, GpuIdentity const &b) noexcept { return not(a == b); }

} // namespace tc
