#include <tc/devices.hpp>

#include <functional>

#include <tc/cpu/cpu_device.hpp>
#if TC_HAS_OPENGL
#include <tc/opengl/opengl_device.hpp>
#endif
#if TC_HAS_VULKAN
#include <tc/vulkan/vulkan_device.hpp>
#endif

namespace tc {

namespace {

struct BackendEntry {
    std::string name;
    std::function<std::unique_ptr<Device>(DeviceRequest const &)> create;
};

std::vector<BackendEntry> backendTable() {
    auto table = std::vector<BackendEntry>{};
    table.push_back({"cpu", [](DeviceRequest const &) { return cpu::createDevice(); }});
#if TC_HAS_OPENGL
#if !defined(__ANDROID__)
    table.push_back({"opengl", [](DeviceRequest const &request) {
                         auto config = opengl::OpenGLDeviceConfig{};
                         config.api = opengl::OpenGLApi::Desktop;
                         config.deviceNameContains = request.deviceNameContains;
                         return opengl::createDevice(config);
                     }});
#endif
    table.push_back({"gles", [](DeviceRequest const &request) {
                         auto config = opengl::OpenGLDeviceConfig{};
                         config.api = opengl::OpenGLApi::ES;
                         config.deviceNameContains = request.deviceNameContains;
                         return opengl::createDevice(config);
                     }});
#endif
#if TC_HAS_VULKAN
    table.push_back({"vulkan", [](DeviceRequest const &request) {
                         auto config = vulkan::VulkanDeviceConfig{};
                         config.deviceNameContains = request.deviceNameContains;
                         return vulkan::createDevice(config);
                     }});
#endif
    return table;
}

} // namespace

std::vector<std::string> availableBackends() {
    auto names = std::vector<std::string>{};
    for (auto const &entry : backendTable()) {
        names.push_back(entry.name);
    }
    return names;
}

std::unique_ptr<Device> createDevice(DeviceRequest const &request) {
    for (auto const &entry : backendTable()) {
        if (entry.name == request.backend) {
            return entry.create(request);
        }
    }
    throw DeviceError{"unknown backend '" + request.backend + "'"};
}

} // namespace tc
