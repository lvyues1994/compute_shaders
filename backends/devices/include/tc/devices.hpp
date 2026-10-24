#pragma once

#include <memory>
#include <string>
#include <vector>

#include <tc/device.hpp>

namespace tc {

struct DeviceRequest {
    std::string backend = "cpu";    // one of availableBackends()
    std::string deviceNameContains; // optional GPU name filter, e.g. "llvmpipe"
};

// Backends compiled into this build: "cpu", plus "opengl", "gles" and
// "vulkan" when enabled. "opengl" is unavailable on Android.
std::vector<std::string> availableBackends();

// Creates a device by backend name; throws DeviceError when the backend is
// unknown or no matching device can be initialised.
std::unique_ptr<Device> createDevice(DeviceRequest const &request);

} // namespace tc
