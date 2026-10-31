#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <tc/buffer.hpp>
#include <tc/mat.hpp>
#include <tc/span.hpp>
#include <tc/vec.hpp>

// Ray queries (GL_EXT_ray_query): kernels trace rays through acceleration
// structures the device builds (Device::createBottomLevel, createTopLevel,
// buildTopLevel). The functions and constants have their GLSL names. GPU
// devices run them where DeviceLimits::features has KernelFeature::RayQuery
// (Vulkan with VK_KHR_ray_query); the CPU backend traverses a BVH it builds
// on the host.
namespace tc {

// Ray flags (rayQueryInitializeEXT).
inline constexpr uint gl_RayFlagsNoneEXT = 0u;
inline constexpr uint gl_RayFlagsOpaqueEXT = 1u;
inline constexpr uint gl_RayFlagsNoOpaqueEXT = 2u;
inline constexpr uint gl_RayFlagsTerminateOnFirstHitEXT = 4u;
inline constexpr uint gl_RayFlagsSkipClosestHitShaderEXT = 8u; // no effect on ray queries
inline constexpr uint gl_RayFlagsCullBackFacingTrianglesEXT = 16u;
inline constexpr uint gl_RayFlagsCullFrontFacingTrianglesEXT = 32u;
inline constexpr uint gl_RayFlagsCullOpaqueEXT = 64u;
inline constexpr uint gl_RayFlagsCullNoOpaqueEXT = 128u;

// rayQueryGetIntersectionTypeEXT(q, false) and (q, true).
inline constexpr uint gl_RayQueryCandidateIntersectionTriangleEXT = 0u;
inline constexpr uint gl_RayQueryCandidateIntersectionAABBEXT = 1u;
inline constexpr uint gl_RayQueryCommittedIntersectionNoneEXT = 0u;
inline constexpr uint gl_RayQueryCommittedIntersectionTriangleEXT = 1u;
inline constexpr uint gl_RayQueryCommittedIntersectionGeneratedEXT = 2u;

// Flags of a TopLevelInstance (VkGeometryInstanceFlagBitsKHR), host side.
enum InstanceFlag : std::uint32_t {
    instanceTriangleFacingCullDisable = 1u,
    instanceTriangleFlipFacing = 2u,
    instanceForceOpaque = 4u,
    instanceForceNoOpaque = 8u,
};

// One instance of a top-level acceleration structure, laid out like
// VkAccelerationStructureInstanceKHR (64 bytes), so a kernel can write it
// through a BufferBinding<TopLevelInstance> and the structure is built
// from the device's copy (Device::buildTopLevel).
struct TopLevelInstance {
    std::array<vec4, 3> transform{}; // rows of the 3x4 object-to-world matrix
    uint customIndexAndMask = 0;     // custom index in bits 0-23, visibility mask in bits 24-31
    uint offsetAndFlags = 0;         // binding table offset (unused) in bits 0-23, InstanceFlag bits in 24-31
    uvec2 reference{};               // BottomLevelAS::reference() of the instanced geometry
};

// Host side: an instance of `blas` (BottomLevelAS::reference()) with the
// object-to-world transform `objectToWorld` (its last row is ignored).
inline TopLevelInstance makeInstance(mat4 const &objectToWorld, uvec2 const reference, uint const customIndex,
                                     uint const mask = 0xFFu, uint const flags = 0u) noexcept {
    auto instance = TopLevelInstance{};
    for (int row = 0; row < 3; ++row) {
        instance.transform[static_cast<std::size_t>(row)] =
            vec4{objectToWorld[0][row], objectToWorld[1][row], objectToWorld[2][row], objectToWorld[3][row]};
    }
    instance.customIndexAndMask = (customIndex & 0xFFFFFFu) | (mask << 24);
    instance.offsetAndFlags = flags << 24;
    instance.reference = reference;
    return instance;
}

// How the device builds an acceleration structure.
struct BuildOptions {
    bool preferFastBuild = false; // else fast traversal
};

namespace detail {

// What the copies of one acceleration structure share.
struct AccelerationState {
    AccelerationState() : identity(ResourceIdentity::create()) {}

    ResourceIdentity identity;
    std::uint64_t reference = 0; // a bottom level's device address (CPU backend: its number)
    std::uint32_t count = 0;     // triangles of a bottom level, the capacity of a top level
    // The CPU backend's structure: a BottomLevelHost or TopLevelHost.
    std::shared_ptr<void const> host;
};

// The CPU backend's BVH: a node is a leaf of `count` items from `first`,
// or has children `first` and `first + 1`.
struct BvhNode {
    vec3 low;
    vec3 high;
    std::uint32_t first = 0;
    std::uint32_t count = 0;
};

struct HostTriangle {
    vec3 a;
    vec3 b;
    vec3 c;
    std::uint32_t primitive = 0;
    std::uint32_t geometry = 0;
    bool opaque = true;
};

struct BottomLevelHost {
    std::vector<BvhNode> nodes;
    std::vector<HostTriangle> triangles;
};

struct HostInstance {
    std::array<vec4, 3> objectToWorld{};
    std::array<vec4, 3> worldToObject{};
    std::uint32_t customIndex = 0;
    std::uint32_t mask = 0;
    std::uint32_t flags = 0;
    std::uint32_t index = 0; // in the instance buffer
    std::shared_ptr<BottomLevelHost const> blas;
};

struct TopLevelHost {
    std::vector<BvhNode> nodes;
    std::vector<HostInstance> instances;
};

// The input of one geometry, read from the host copies of its buffers.
struct HostGeometry {
    ConstByteSpan positions; // from vertex 0
    std::uint32_t stride = 16;
    std::uint32_t vertexCount = 0;
    ConstByteSpan indices; // from the first index, three per triangle
    std::uint32_t triangleCount = 0;
    bool opaque = true;
};

// Binned-SAH builds (tc/src/ray_query.cpp). buildTopLevel returns nullopt
// with the index of an instance whose bottom level `lookup` cannot find.
std::shared_ptr<BottomLevelHost const> buildBottomLevel(std::vector<HostGeometry> const &geometries);
std::shared_ptr<TopLevelHost const> buildTopLevel(std::vector<HostInstance> instances);

} // namespace detail

// A bottom-level acceleration structure: triangles of one or more
// geometries (Device::createBottomLevel). Copies refer to the same one; it
// is freed after the last copy, once the device no longer uses it. Top
// levels refer to it by reference() and do not keep it alive: it must live
// as long as a top level built with it may be traced.
struct BottomLevelAS {
    BottomLevelAS() = default;

    explicit operator bool() const noexcept { return state != nullptr; }
    // What TopLevelInstance::reference holds: the device address on GPUs.
    uvec2 reference() const noexcept {
        return uvec2{static_cast<uint>(state->reference & 0xFFFFFFFFu), static_cast<uint>(state->reference >> 32)};
    }
    std::uint32_t triangleCount() const noexcept { return state->count; }
    ResourceIdentity const &identity() const noexcept { return state->identity; }
    std::shared_ptr<detail::AccelerationState> const &shared() const noexcept { return state; }

private:
    friend struct Device;
    explicit BottomLevelAS(std::shared_ptr<detail::AccelerationState> state_) noexcept : state(std::move(state_)) {}

    std::shared_ptr<detail::AccelerationState> state;
};

// A top-level acceleration structure of up to capacity() instances
// (Device::createTopLevel), rebuilt from an instance buffer with
// Device::buildTopLevel. Kernels trace it through AccelerationStructureBinding.
struct TopLevelAS {
    TopLevelAS() = default;

    explicit operator bool() const noexcept { return state != nullptr; }
    std::uint32_t capacity() const noexcept { return state->count; }
    ResourceIdentity const &identity() const noexcept { return state->identity; }
    std::shared_ptr<detail::AccelerationState> const &shared() const noexcept { return state; }

private:
    friend struct Device;
    explicit TopLevelAS(std::shared_ptr<detail::AccelerationState> state_) noexcept : state(std::move(state_)) {}

    std::shared_ptr<detail::AccelerationState> state;
};

// `layout(binding = Binding) uniform accelerationStructureEXT name;`
template <std::uint32_t Binding>
struct AccelerationStructureBinding {
    static constexpr std::uint32_t binding = Binding;

    void attach(TopLevelAS const &structure_) noexcept { structure = structure_.shared().get(); }
    detail::AccelerationState const *attachedStructure() const noexcept { return structure; }

private:
    detail::AccelerationState const *structure = nullptr;
};

namespace detail {

struct RayHit {
    float t = 0.0f;
    vec2 barycentrics;
    bool frontFace = false;
    bool opaque = true;
    std::uint32_t instance = 0; // in TopLevelHost::instances
    std::uint32_t primitive = 0;
    std::uint32_t geometry = 0;
};

// What the CPU traversal has still to look at, nearest first.
struct RayEntry {
    float t = 0.0f;
    std::uint32_t kind = 0;     // 0: a top-level node, 1: a bottom-level node, 2: a hit
    std::uint32_t node = 0;     // or the hit's index in RayQueryState::hits
    std::uint32_t instance = 0; // of a bottom-level node
};

} // namespace detail

// rayQueryEXT: the state of one ray query. Only a local variable of a
// kernel function; it cannot be copied into buffers or structs.
struct rayQueryEXT {
    detail::TopLevelHost const *scene = nullptr;
    uint flags = 0;
    uint cullMask = 0;
    vec3 origin;
    vec3 direction;
    float tMin = 0.0f;
    float tMax = 0.0f;
    bool hasCommitted = false;
    bool terminated = false;
    detail::RayHit committed;
    detail::RayHit candidate;
    std::vector<detail::RayEntry> heap;
    std::vector<detail::RayHit> hits;
};

namespace detail {

// Traversal of the CPU backend (tc/src/ray_query.cpp).
void rayQueryStart(rayQueryEXT &query);
bool rayQueryStep(rayQueryEXT &query);
HostInstance const &rayQueryInstance(rayQueryEXT const &query, bool committed);

} // namespace detail

template <std::uint32_t B>
void rayQueryInitializeEXT(rayQueryEXT &query, AccelerationStructureBinding<B> const &topLevel, uint const rayFlags,
                           uint const cullMask, vec3 const &origin, float const tMin, vec3 const &direction,
                           float const tMax) {
    auto const *structure = topLevel.attachedStructure();
    assert(structure != nullptr and structure->host != nullptr and "trace a built TopLevelAS");
    query.scene = static_cast<detail::TopLevelHost const *>(structure->host.get());
    query.flags = rayFlags;
    query.cullMask = cullMask & 0xFFu;
    query.origin = origin;
    query.direction = direction;
    query.tMin = tMin;
    query.tMax = tMax;
    detail::rayQueryStart(query);
}

// Continues the traversal; true when it stops at a candidate that is not
// opaque, for the kernel to confirm or not.
inline bool rayQueryProceedEXT(rayQueryEXT &query) { return detail::rayQueryStep(query); }

inline void rayQueryTerminateEXT(rayQueryEXT &query) {
    query.terminated = true;
    query.heap.clear();
}

// Commits the current candidate.
inline void rayQueryConfirmIntersectionEXT(rayQueryEXT &query) {
    query.committed = query.candidate;
    query.hasCommitted = true;
    if ((query.flags & gl_RayFlagsTerminateOnFirstHitEXT) != 0u) {
        rayQueryTerminateEXT(query);
    }
}

inline uint rayQueryGetIntersectionTypeEXT(rayQueryEXT const &query, bool const committed) {
    if (committed) {
        return query.hasCommitted ? gl_RayQueryCommittedIntersectionTriangleEXT : gl_RayQueryCommittedIntersectionNoneEXT;
    }
    return gl_RayQueryCandidateIntersectionTriangleEXT;
}

inline float rayQueryGetIntersectionTEXT(rayQueryEXT const &query, bool const committed) {
    return committed ? query.committed.t : query.candidate.t;
}

inline vec2 rayQueryGetIntersectionBarycentricsEXT(rayQueryEXT const &query, bool const committed) {
    return committed ? query.committed.barycentrics : query.candidate.barycentrics;
}

inline bool rayQueryGetIntersectionFrontFaceEXT(rayQueryEXT const &query, bool const committed) {
    return committed ? query.committed.frontFace : query.candidate.frontFace;
}

inline int rayQueryGetIntersectionInstanceCustomIndexEXT(rayQueryEXT const &query, bool const committed) {
    return static_cast<int>(detail::rayQueryInstance(query, committed).customIndex);
}

inline int rayQueryGetIntersectionInstanceIdEXT(rayQueryEXT const &query, bool const committed) {
    return static_cast<int>(detail::rayQueryInstance(query, committed).index);
}

inline int rayQueryGetIntersectionPrimitiveIndexEXT(rayQueryEXT const &query, bool const committed) {
    return static_cast<int>(committed ? query.committed.primitive : query.candidate.primitive);
}

inline int rayQueryGetIntersectionGeometryIndexEXT(rayQueryEXT const &query, bool const committed) {
    return static_cast<int>(committed ? query.committed.geometry : query.candidate.geometry);
}

inline mat4x3 rayQueryGetIntersectionObjectToWorldEXT(rayQueryEXT const &query, bool const committed) {
    auto const &rows = detail::rayQueryInstance(query, committed).objectToWorld;
    return mat4x3{vec3{rows[0].x, rows[1].x, rows[2].x}, vec3{rows[0].y, rows[1].y, rows[2].y},
                  vec3{rows[0].z, rows[1].z, rows[2].z}, vec3{rows[0].w, rows[1].w, rows[2].w}};
}

inline mat4x3 rayQueryGetIntersectionWorldToObjectEXT(rayQueryEXT const &query, bool const committed) {
    auto const &rows = detail::rayQueryInstance(query, committed).worldToObject;
    return mat4x3{vec3{rows[0].x, rows[1].x, rows[2].x}, vec3{rows[0].y, rows[1].y, rows[2].y},
                  vec3{rows[0].z, rows[1].z, rows[2].z}, vec3{rows[0].w, rows[1].w, rows[2].w}};
}

} // namespace tc
