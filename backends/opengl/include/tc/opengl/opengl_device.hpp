#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <tc/device.hpp>

namespace tc::opengl {

enum class OpenGLApi : std::uint8_t {
    Desktop, // OpenGL 4.3 core, runs the `#version 430` GLSL
    ES,      // OpenGL ES 3.1 (Android, embedded), runs the `#version 310 es` GLSL
};

struct OpenGLDeviceConfig {
    OpenGLApi api = OpenGLApi::Desktop;
    std::string deviceNameContains; // match against GL_RENDERER; empty picks the first GPU
    bool debugOutput = false;       // print KHR_debug messages to stderr
};

// Creates a headless EGL context. The device must be used from the thread
// that created it.
std::unique_ptr<Device> createDevice(OpenGLDeviceConfig const &config = {});

} // namespace tc::opengl
