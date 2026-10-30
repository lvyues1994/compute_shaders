#pragma once

#include <cstdint>
#include <type_traits>

#include <tc/tc.hpp>

// Kernels for kernel templates, indirect dispatch and buffer ranges.
namespace tests {

using namespace tc;

// Two material models plugged into one kernel template.
struct Lambert {
    static constexpr float kAmbient = 0.1f;
    static float shade(float const nDotL) { return max(nDotL, 0.0f); }
};

struct Toon {
    static constexpr float kAmbient = 0.2f;
    static constexpr int kBands = 3;
    static float shade(float const nDotL) { return floor(max(nDotL, 0.0f) * float(kBands)) / float(kBands); }
};

template <class Material, uint GroupSize = 32>
struct TC_KERNEL Shade {
    static constexpr char const *fileLocation = tc::instanceName<Shade>();
    uvec3 local_size{GroupSize, 1, 1};

    static constexpr float kLift = Material::kAmbient * 2.0f;

    BufferBinding<float, 0> nDotL;
    BufferBinding<vec2, 1> out;
    Uniform<float, 0> exposure{};

    float tone(float const x) const { return x * exposure.value + kLift; }

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(nDotL.length())) {
            return;
        }
        float s = Material::shade(nDotL[i]);
        if constexpr (std::is_same_v<Material, Toon>) {
            s += 0.01f;
        }
        out[i] = vec2(tone(s), float(gl_WorkGroupSize.x));
    }
};

TC_KERNEL_INSTANCE("shade_lambert", Shade<Lambert>);
TC_KERNEL_INSTANCE("shade_toon", Shade<Toon, 64>);

// GPU culling with compaction: visible spheres get consecutive slots, and
// every 64th slot adds one workgroup to the indirect dispatch that shades
// them.
struct TC_KERNEL CullSpheres {
    static constexpr char fileLocation[] = "cull_spheres";
    uvec3 local_size{64, 1, 1};

    BufferBinding<vec4, 0> spheres; // centre, radius
    BufferBinding<uint, 1> visible;
    BufferBinding<uint, 2> count;
    BufferBinding<DispatchIndirectCommand, 3> args;
    Uniform<vec4, 0> plane{}; // inside when dot(normal, centre) + d > -radius

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(spheres.length())) {
            return;
        }
        vec4 s = spheres[i];
        if (dot(plane.value.xyz(), s.xyz()) + plane.value.w > -s.w) {
            uint slot = atomicAdd(count[0], 1u);
            visible[slot] = i;
            if (slot % 64u == 0u) {
                atomicAdd(args[0].x, 1u);
            }
        }
    }
};

struct TC_KERNEL ShadeVisible {
    static constexpr char fileLocation[] = "shade_visible";
    uvec3 local_size{64, 1, 1};

    BufferBinding<vec4, 0> spheres;
    BufferBinding<uint, 1> visible;
    BufferBinding<uint, 2> count;
    BufferBinding<float, 3> out;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= count[0]) {
            return;
        }
        uint index = visible[i];
        out[index] = spheres[index].w * 2.0f + float(gl_NumWorkGroups.x);
    }
};

// Reads and writes buffer ranges: element 0 is the first element of the
// bound range, length() its size.
struct TC_KERNEL ScaleRange {
    static constexpr char fileLocation[] = "scale_range";
    uvec3 local_size{64, 1, 1};

    BufferBinding<vec4, 0> in;
    BufferBinding<vec4, 1> out;
    Uniform<float, 0> factor{};

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(in.length())) {
            return;
        }
        out[i] = in[i] * factor.value + vec4(float(in.length()));
    }
};

struct Body {
    vec3 position;
    float mass;
    uint id;
};

struct TC_KERNEL CopyBodies {
    static constexpr char fileLocation[] = "copy_bodies";
    uvec3 local_size{64, 1, 1};

    TableBinding<Body, 0, Soa<16>> bodies;
    BufferBinding<vec4, 1> out;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(bodies.length())) {
            return;
        }
        out[i] = vec4(bodies[i].position, bodies[i].mass + float(bodies[i].id));
    }
};

// Shading arithmetic at medium precision; positions stay full precision in
// the buffer.
struct TC_KERNEL MediumPrecisionShading {
    static constexpr char fileLocation[] = "medium_precision_shading";
    static constexpr Precision floatPrecision = Precision::Medium;
    uvec3 local_size{64, 1, 1};

    BufferBinding<vec4, 0> normals;
    BufferBinding<vec4, 1> shaded;
    Uniform<vec3, 0> light{};

    vec3 tonemap(vec3 const c) const { return c / (c + vec3(1.0f)); }

    void main() {
        uint i = gl_GlobalInvocationID.x;
        vec3 n = normalize(normals[i].xyz());
        float diffuse = max(dot(n, normalize(light.value)), 0.0f);
        float rim = pow(1.0f - abs(n.z), 3.0f);
        shaded[i] = vec4(tonemap(vec3(diffuse * 2.0f, rim, diffuse * rim)), normals[i].w);
    }
};

// The reload test edits this kernel's generated GLSL on disk.
struct TC_KERNEL ReloadMe {
    static constexpr char fileLocation[] = "reload_me";
    uvec3 local_size{64, 1, 1};

    BufferBinding<float, 0> out;

    void main() { out[gl_GlobalInvocationID.x] = 1.0f; }
};

// Every invocation records its position in a 3D grid of workgroups.
struct TC_KERNEL GridIds {
    static constexpr char fileLocation[] = "grid_ids";
    uvec3 local_size{4, 2, 1};

    BufferBinding<uvec4, 0> out;

    void main() {
        uvec3 n = gl_NumWorkGroups * gl_WorkGroupSize;
        uvec3 g = gl_GlobalInvocationID;
        out[g.x + n.x * (g.y + n.y * g.z)] = uvec4(g, gl_NumWorkGroups.x * 100u + gl_NumWorkGroups.y * 10u + gl_NumWorkGroups.z);
    }
};

} // namespace tests
