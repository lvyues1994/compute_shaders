#pragma once

#include <memory>
#include <string>

#include <tc/device.hpp>

namespace tc::vulkan {

struct VulkanDeviceConfig {
    std::string deviceNameContains; // match against the physical device name; empty prefers a discrete GPU
    bool enableValidation = false;  // enable VK_LAYER_KHRONOS_validation when installed
};

// Runs the transpiled SPIR-V. Every transfer and dispatch is submitted and
// waited for before the call returns.
std::unique_ptr<Device> createDevice(VulkanDeviceConfig const &config = {});

} // namespace tc::vulkan
