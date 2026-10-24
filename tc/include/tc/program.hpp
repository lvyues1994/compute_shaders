#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_map>

#include <tc/dim.hpp>
#include <tc/gpu_format.hpp>
#include <tc/span.hpp>
#include <tc/vec.hpp>

namespace tc {

enum class ResourceKind : std::uint8_t {
    StorageBuffer,
    StorageImage,
};

enum class Access : std::uint8_t {
    ReadWrite,
    ReadOnly,
    WriteOnly,
};

enum class ScalarKind : std::uint8_t {
    Int,
    UInt,
    Float,
    Double,
};

// Reflection of one buffer or image binding of a transpiled kernel.
struct ResourceSlot {
    std::string_view name;
    std::uint32_t binding = 0;
    ResourceKind kind = ResourceKind::StorageBuffer;
    Access access = Access::ReadWrite;
    Dim imageDim = Dim::D2;
    ImageFormat imageFormat{};
};

// Reflection of one Uniform<T, Location> member.
struct UniformSlot {
    std::string_view name;
    std::uint32_t location = 0;
    ScalarKind scalar = ScalarKind::Float;
    std::uint32_t components = 1;
    std::uint32_t pushConstantOffset = 0;
};

// Everything the transpiler produced for one kernel struct.
struct KernelProgram {
    std::string_view name;
    uvec3 localSize{};
    std::string_view glslDesktop;          // `#version 430`, OpenGL
    std::string_view glslEs;               // `#version 310 es`, OpenGL ES; empty if unsupported
    std::string_view esUnavailableReason;  // why glslEs is empty
    Span<std::uint32_t const> spirv;       // Vulkan SPIR-V
    Span<ResourceSlot const> resources;
    Span<UniformSlot const> uniforms;
    std::uint32_t pushConstantSize = 0;
};

// Process-wide table of transpiled kernels, filled by the generated
// registration sources before main() runs and looked up by fileLocation.
struct ProgramRegistry {
    static ProgramRegistry &instance();

    void add(KernelProgram const &program);
    std::optional<KernelProgram> find(std::string_view name) const;

    ProgramRegistry(ProgramRegistry const &) = delete;
    ProgramRegistry &operator=(ProgramRegistry const &) = delete;

private:
    ProgramRegistry() = default;

    mutable std::mutex mutex;
    std::unordered_map<std::string_view, KernelProgram> programs;
};

struct ProgramRegistration {
    explicit ProgramRegistration(KernelProgram const &program) { ProgramRegistry::instance().add(program); }
};

} // namespace tc
