#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <tc/dim.hpp>
#include <tc/gpu_format.hpp>
#include <tc/span.hpp>
#include <tc/texture.hpp>
#include <tc/vec.hpp>

namespace tc {

enum class ResourceKind : std::uint8_t {
    StorageBuffer,
    StorageImage,
    SampledTexture,
    ExternalTexture,
    ExternalImage,         // a platform frame bound as a storage image
    AccelerationStructure, // a top-level acceleration structure for ray queries
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
    Filter filter = Filter::Nearest;   // sampled textures only
    Wrap wrap = Wrap::ClampToEdge;     // sampled textures only
    Mip mip = Mip::None;               // sampled textures only
};

// Reflection of one Uniform<T, Location> member. Matrices have
// `columns` > 1, each column a vector of `components` scalars.
struct UniformSlot {
    std::string_view name;
    std::uint32_t location = 0;
    ScalarKind scalar = ScalarKind::Float;
    std::uint32_t components = 1;
    std::uint32_t columns = 1;
    std::uint32_t pushConstantOffset = 0;
};

// Optional device features a kernel uses (bit flags).
enum class KernelFeature : std::uint32_t {
    Double = 1u << 0,        // double precision arithmetic
    Subgroups = 1u << 1,     // subgroup built-ins
    Int64 = 1u << 2,         // 64-bit integer arithmetic
    Int64Atomics = 1u << 3,  // 64-bit atomics on buffer elements
    RayQuery = 1u << 4,      // ray queries through acceleration structures
};

constexpr std::uint32_t operator|(KernelFeature const a, KernelFeature const b) noexcept {
    return static_cast<std::uint32_t>(a) | static_cast<std::uint32_t>(b);
}

constexpr bool uses(std::uint32_t const features, KernelFeature const feature) noexcept {
    return (features & static_cast<std::uint32_t>(feature)) != 0;
}

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
    std::uint32_t sharedBytes = 0;         // Shared variables of one workgroup
    std::uint32_t features = 0;            // KernelFeature bits
    std::uint32_t revision = 0;            // 0 as compiled in; each ProgramRegistry::reloadFrom adds one
};

// Everything host code compiled against a program relies on, as canonical
// text: local size, resources (with their inferred access), uniforms, push
// constants, shared memory and features. tc-transpile writes it next to the
// GLSL as <name>.interface.
std::string describeInterface(KernelProgram const &program);

struct ReloadReport {
    struct Rejection {
        std::string program;
        std::string reason;
    };
    std::vector<std::string> reloaded;
    std::vector<Rejection> rejected;
};

// Process-wide table of transpiled kernels, filled by the generated
// registration sources before main() runs and looked up by fileLocation.
struct ProgramRegistry {
    static ProgramRegistry &instance();

    void add(KernelProgram const &program);
    std::optional<KernelProgram> find(std::string_view name) const;
    // Every registered program, e.g. to check them all against a device at
    // startup (Device::checkKernel).
    std::vector<KernelProgram> all() const;

    // Development: replaces registered programs with what tc-transpile last
    // wrote to `directory` (the glsl directory of tc_add_kernels), so GPU
    // devices run edited kernel bodies at their next useKernel() without a
    // restart. A program is replaced only if its interface is unchanged:
    // new bindings, uniforms, a changed local size or access need a rebuild,
    // because host code was compiled against them. The CPU backend keeps
    // running the C++ it was compiled from. Builds with TC_PROGRAM_RELOAD
    // off (release builds) never read the disk and report that instead.
    ReloadReport reloadFrom(std::string const &directory);

    ProgramRegistry(ProgramRegistry const &) = delete;
    ProgramRegistry &operator=(ProgramRegistry const &) = delete;

private:
    ProgramRegistry() = default;

    // Sources of reloaded programs. Kept for good: devices may still hold
    // views of an earlier revision.
    struct ReloadedSources {
        std::string desktop;
        std::string es;
        std::vector<std::uint32_t> spirv;
    };

    mutable std::mutex mutex;
    std::unordered_map<std::string_view, KernelProgram> programs;
    std::vector<std::unique_ptr<ReloadedSources>> reloaded;
};

struct ProgramRegistration {
    explicit ProgramRegistration(KernelProgram const &program) { ProgramRegistry::instance().add(program); }
};

} // namespace tc
