#pragma once

#include <string>

#include "kernel_model.hpp"

namespace tc::transpiler {

struct ShaderVariants {
    std::string desktop;             // #version 430, OpenGL
    std::string es;                  // #version 310 es, OpenGL ES; empty when unavailable
    std::string vulkan;              // #version 450, uniforms as push constants
    std::string esUnavailableReason; // set when `es` is empty
};

// Wraps the shared body with the declarations each GLSL dialect needs.
ShaderVariants assembleShaders(KernelModel const &kernel, GlslBody const &body, std::string const &sourceName);

} // namespace tc::transpiler
