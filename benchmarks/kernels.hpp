#pragma once

#include <array>

#include <tc/tc.hpp>

// Kernels measured by tc_bench, alongside the nv12_to_rgba and game_of_life
// examples. Each one stresses one kind of memory traffic.
namespace bench {

using namespace tc;

// Index of this invocation in a 1D problem dispatched as rows of
// gl_NumWorkGroups.x workgroups (more than 65535 groups need several rows).
inline uint linearIndex() {
    return gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x;
}

// Streaming read and write: the bandwidth ceiling.
struct TC_KERNEL CopyVec4 {
    static constexpr char fileLocation[] = "bench_copy_vec4";
    uvec3 local_size{256, 1, 1};

    BufferBinding<vec4, 0> source;
    BufferBinding<vec4, 1> target;

    void main() {
        uint i = linearIndex();
        if (i >= uint(target.length())) {
            return;
        }
        target[i] = source[i];
    }
};

struct TC_KERNEL Saxpy {
    static constexpr char fileLocation[] = "bench_saxpy";
    uvec3 local_size{256, 1, 1};

    BufferBinding<float, 0> x;
    BufferBinding<float, 1> y;
    Uniform<float, 0> a{};

    void main() {
        uint i = linearIndex();
        if (i >= uint(y.length())) {
            return;
        }
        y[i] = a.value * x[i] + y[i];
    }
};

// Workgroup sums through a shared-memory tree.
struct TC_KERNEL ReduceShared {
    static constexpr char fileLocation[] = "bench_reduce_shared";
    static constexpr bool usesBarrier = true;
    uvec3 local_size{256, 1, 1};

    BufferBinding<float, 0> values;
    BufferBinding<float, 1> sums;
    Shared<std::array<float, 256>> partial;

    void main() {
        uint local = gl_LocalInvocationIndex;
        uint i = linearIndex();
        partial[local] = i < uint(values.length()) ? values[i] : 0.0f;
        barrier();
        for (uint stride = 128u; stride > 0u; stride >>= 1u) {
            if (local < stride) {
                partial[local] += partial[local + stride];
            }
            barrier();
        }
        if (local == 0u) {
            sums[gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x] = partial[0];
        }
    }
};

// Reads rows, writes columns: every write lands in a different cache line.
struct TC_KERNEL TransposeNaive {
    static constexpr char fileLocation[] = "bench_transpose_naive";
    uvec3 local_size{16, 16, 1};

    BufferBinding<float, 0> input;
    BufferBinding<float, 1> output;
    Uniform<ivec2, 0> size{};

    void main() {
        ivec2 p = ivec2(gl_GlobalInvocationID.xy());
        if (p.x < size.value.x and p.y < size.value.y) {
            output[p.x * size.value.y + p.y] = input[p.y * size.value.x + p.x];
        }
    }
};

// The same transpose through a padded shared tile: both sides coalesced.
struct TC_KERNEL TransposeTiled {
    static constexpr char fileLocation[] = "bench_transpose_tiled";
    static constexpr bool usesBarrier = true;
    uvec3 local_size{16, 16, 1};

    BufferBinding<float, 0> input;
    BufferBinding<float, 1> output;
    Uniform<ivec2, 0> size{};
    Shared<std::array<std::array<float, 17>, 16>> tile;

    void main() {
        ivec2 local = ivec2(gl_LocalInvocationID.xy());
        ivec2 origin = ivec2(gl_WorkGroupID.xy()) * 16;
        ivec2 p = origin + local;
        if (p.x < size.value.x and p.y < size.value.y) {
            tile[local.y][local.x] = input[p.y * size.value.x + p.x];
        }
        barrier();
        ivec2 q = ivec2{origin.y, origin.x} + local;
        if (q.x < size.value.y and q.y < size.value.x) {
            output[q.y * size.value.y + q.x] = tile[local.x][local.y];
        }
    }
};

// A particle in the only shape std430 and C++ agree on without padding
// tricks: mass and age ride in the w components.
struct Particle {
    vec4 positionMass;
    vec4 velocityAge;
};

struct TC_KERNEL ParticlesStd430 {
    static constexpr char fileLocation[] = "bench_particles_std430";
    uvec3 local_size{256, 1, 1};

    BufferBinding<Particle, 0> particles;
    Uniform<float, 0> dt{};

    void main() {
        uint i = linearIndex();
        if (i >= uint(particles.length())) {
            return;
        }
        Particle p = particles[i];
        vec3 velocity = p.velocityAge.xyz() + vec3{0.0f, -9.81f, 0.0f} * dt.value;
        vec3 position = p.positionMass.xyz() + velocity * dt.value;
        p.positionMass = vec4(position, p.positionMass.w);
        p.velocityAge = vec4(velocity, p.velocityAge.w + dt.value);
        particles[i] = p;
    }
};

// The same particle in its natural shape, stored in a packed table: also 8
// words, so this measures the cost of unpacking.
struct ParticleData {
    vec3 position;
    float mass;
    vec3 velocity;
    float age;
};

// Kernels that differ only in the table layout share their main().
#define TC_BENCH_PARTICLES_MAIN                                                                                        \
    void main() {                                                                                                      \
        uint i = linearIndex();                                                                                        \
        if (i >= uint(particles.length())) {                                                                           \
            return;                                                                                                    \
        }                                                                                                              \
        ParticleData p = particles[i];                                                                                 \
        p.velocity += vec3{0.0f, -9.81f, 0.0f} * dt.value;                                                             \
        p.position += p.velocity * dt.value;                                                                           \
        p.age += dt.value;                                                                                             \
        particles[i] = p;                                                                                              \
    }

struct TC_KERNEL ParticlesTable {
    static constexpr char fileLocation[] = "bench_particles_table";
    uvec3 local_size{256, 1, 1};

    TableBinding<ParticleData, 0> particles;
    Uniform<float, 0> dt{};

    TC_BENCH_PARTICLES_MAIN
};

struct TC_KERNEL ParticlesSoa {
    static constexpr char fileLocation[] = "bench_particles_soa";
    uvec3 local_size{256, 1, 1};

    TableBinding<ParticleData, 0, Soa<>> particles;
    Uniform<float, 0> dt{};

    TC_BENCH_PARTICLES_MAIN
};

#undef TC_BENCH_PARTICLES_MAIN

// A scene instance: 21 words. Culling reads `flags` of every instance and
// the rest only for the few that are enabled, the access pattern where the
// layout matters most.
struct Instance {
    mat3 rotation;
    vec3 translation;
    vec4 color;
    vec3 center;
    float radius;
    uint flags; // bit 0: enabled
};

#define TC_BENCH_CULL_MAIN                                                                                             \
    void main() {                                                                                                      \
        uint i = linearIndex();                                                                                        \
        if (i >= uint(instances.length()) or (instances[i].flags & 1u) == 0u) {                                        \
            return;                                                                                                    \
        }                                                                                                              \
        vec3 world = instances[i].rotation * instances[i].center + instances[i].translation;                           \
        float light = max(dot(normalize(world), vec3{0.0f, 1.0f, 0.0f}), 0.0f) * instances[i].radius;                  \
        visible[i] = instances[i].color * light;                                                                       \
    }

struct TC_KERNEL CullAos {
    static constexpr char fileLocation[] = "bench_cull_aos";
    uvec3 local_size{256, 1, 1};

    TableBinding<Instance, 0> instances;
    BufferBinding<vec4, 1> visible;

    TC_BENCH_CULL_MAIN
};

struct TC_KERNEL CullSoa {
    static constexpr char fileLocation[] = "bench_cull_soa";
    uvec3 local_size{256, 1, 1};

    TableBinding<Instance, 0, Soa<>> instances;
    BufferBinding<vec4, 1> visible;

    TC_BENCH_CULL_MAIN
};

#undef TC_BENCH_CULL_MAIN

// Per-instance shading inputs, read in full: 25 words plain, 15 through
// codecs (affine transform, 8-bit colour, octahedral normal, half uv).
struct ShadingPlain {
    mat4 transform;
    vec4 color;
    vec3 normal;
    vec2 uv;
};

struct ShadingEncoded {
    Encoded<mat4, Affine3x4> transform;
    Encoded<vec4, Unorm8x4> color;
    Encoded<vec3, OctahedralNormal> normal;
    Encoded<vec2, Half2> uv;
};

#define TC_BENCH_SHADE_MAIN                                                                                            \
    void main() {                                                                                                      \
        uint i = linearIndex();                                                                                        \
        if (i >= uint(inputs.length())) {                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
        mat4 transform = inputs[i].transform;                                                                          \
        vec4 color = inputs[i].color;                                                                                  \
        vec3 normal = inputs[i].normal;                                                                                \
        vec2 uv = inputs[i].uv;                                                                                        \
        vec3 world = (transform * vec4(normal, 0.0f)).xyz();                                                           \
        shaded[i] = vec4(world * color.xyz() + vec3(uv, 0.0f), color.w);                                               \
    }

struct TC_KERNEL ShadePlain {
    static constexpr char fileLocation[] = "bench_shade_plain";
    uvec3 local_size{256, 1, 1};

    TableBinding<ShadingPlain, 0> inputs;
    BufferBinding<vec4, 1> shaded;

    TC_BENCH_SHADE_MAIN
};

struct TC_KERNEL ShadeEncoded {
    static constexpr char fileLocation[] = "bench_shade_encoded";
    uvec3 local_size{256, 1, 1};

    TableBinding<ShadingEncoded, 0> inputs;
    BufferBinding<vec4, 1> shaded;

    TC_BENCH_SHADE_MAIN
};

#undef TC_BENCH_SHADE_MAIN

// 3-component points: padded to vec4 in a std430 buffer, 12 bytes in a table.
struct TC_KERNEL ScalePointsVec4 {
    static constexpr char fileLocation[] = "bench_scale_points_vec4";
    uvec3 local_size{256, 1, 1};

    BufferBinding<vec4, 0> points;

    void main() {
        uint i = linearIndex();
        if (i >= uint(points.length())) {
            return;
        }
        points[i] = vec4(points[i].xyz() * 1.5f + vec3(1.0f), 0.0f);
    }
};

struct TC_KERNEL ScalePointsTable {
    static constexpr char fileLocation[] = "bench_scale_points_table";
    uvec3 local_size{256, 1, 1};

    TableBinding<vec3, 0> points;

    void main() {
        uint i = linearIndex();
        if (i >= uint(points.length())) {
            return;
        }
        points[i] = points[i] * 1.5f + vec3(1.0f);
    }
};

} // namespace bench
