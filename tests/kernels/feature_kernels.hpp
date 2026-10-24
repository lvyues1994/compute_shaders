#pragma once

#include <array>
#include <cstdint>

#include <tc/tc.hpp>

// Small kernels that each exercise one area of the transpiler. The
// equivalence tests run them on every backend and compare with the CPU.
namespace tests {

using namespace tc;

struct Item {
    vec3 a;
    float b;
    vec3 c;
    float d;
};

// Constant arrays of structs read with a dynamic index.
struct TC_KERNEL ConstStructArray {
    static constexpr char fileLocation[] = "const_struct_array";
    uvec3 local_size{4, 1, 1};

    BufferBinding<vec4, 0> out;

    static constexpr std::array<Item, 4> kItems{
        Item{vec3{1.0f, 2.0f, 3.0f}, 4.0f, vec3{5.0f, 6.0f, 7.0f}, 8.0f},
        Item{vec3{9.0f, 10.0f, 11.0f}, 12.0f, vec3{13.0f, 14.0f, 15.0f}, 16.0f},
        Item{vec3{17.0f, 18.0f, 19.0f}, 20.0f, vec3{21.0f, 22.0f, 23.0f}, 24.0f},
        Item{vec3{25.0f, 26.0f, 27.0f}, 28.0f, vec3{29.0f, 30.0f, 31.0f}, 32.0f},
    };

    void main() {
        uint i = gl_GlobalInvocationID.x;
        Item item = kItems[i];
        out[2 * i] = vec4(item.a, item.b);
        out[2 * i + 1] = vec4(item.c, item.d);
    }
};

// GLSL built-in functions against their tc:: emulation.
struct TC_KERNEL MathFunctions {
    static constexpr char fileLocation[] = "math_functions";
    uvec3 local_size{64, 1, 1};

    BufferBinding<float, 0> input;
    BufferBinding<vec4, 1> output;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(input.length())) {
            return;
        }
        float x = input[i];
        vec3 v = vec3{x, x * 0.5f, 1.0f - x};
        uint base = i * 4u;
        output[base] = vec4(sin(x), cos(x), sqrt(abs(x)), pow(abs(x) + 1.0f, 1.5f));
        output[base + 1u] = vec4(floor(x * 3.0f), fract(x * 3.0f), mod(x * 7.0f, 2.0f), sign(x - 0.5f));
        output[base + 2u] = vec4(clamp(x, 0.2f, 0.8f), mix(1.0f, 3.0f, x), step(0.5f, x), smoothstep(0.1f, 0.9f, x));
        output[base + 3u] = vec4(normalize(v + vec3{0.1f}), length(v) + dot(v, vec3{1.0f, 2.0f, 3.0f}));
    }
};

// Integer arithmetic, mixed-type conversions and control flow.
struct TC_KERNEL IntegerOps {
    static constexpr char fileLocation[] = "integer_ops";
    uvec3 local_size{32, 1, 1};

    BufferBinding<std::int32_t, 0> ints;
    BufferBinding<uint, 1> uints;

    int collatzSteps(uint start) const {
        uint n = start + 1u;
        int steps = 0;
        while (n != 1u and steps < 200) {
            n = (n % 2u == 0u) ? n / 2u : 3u * n + 1u;
            ++steps;
        }
        return steps;
    }

    void main() {
        uint i = gl_GlobalInvocationID.x;
        int s = int(i) - 16;
        int acc = 0;
        for (int k = 0; k < 5; ++k) {
            if (k == 3) {
                continue;
            }
            acc += s * k - (int(i) % 3) + (s >> 1);
        }
        int mixed = 7;
        mixed += 1.5f * float(s);
        switch (s & 3) {
        case 0:
            acc += 100;
            break;
        case 1:
            acc -= 100;
            break;
        default:
            acc ^= 0x55;
            break;
        }
        bool odd = (i & 1u) != 0u;
        uint bits = (i << 3u) | (i >> 2u) | (odd ? 0x100u : 0u);
        int k = 0;
        do {
            ++k;
        } while (k < s);
        ints[3 * i] = acc;
        ints[3 * i + 1] = mixed;
        ints[3 * i + 2] = k + collatzSteps(i);
        uints[2 * i] = bits ^ ~i;
        uints[2 * i + 1] = uint(s) + uint(odd);
    }
};

struct Particle {
    vec4 position;
    vec4 velocity;
    float mass;
    std::int32_t id;
    vec2 pad;
};

// User structs in a storage buffer: std430 layout must match C++.
struct TC_KERNEL ParticleStep {
    static constexpr char fileLocation[] = "particle_step";
    uvec3 local_size{64, 1, 1};

    BufferBinding<Particle, 0> particles;
    Uniform<float, 0> dt{0.0f};
    Uniform<vec3, 1> gravity{};

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(particles.length())) {
            return;
        }
        Particle p = particles[i];
        p.velocity += vec4(gravity.value * dt, 0.0f) / p.mass;
        p.position += p.velocity * dt;
        p.id += 1000;
        particles[i] = p;
    }
};

// Uniforms of several types: push-constant offsets on Vulkan, glUniform* on GL.
struct TC_KERNEL UniformTypes {
    static constexpr char fileLocation[] = "uniform_types";
    uvec3 local_size{1, 1, 1};

    BufferBinding<vec4, 0> out;
    Uniform<std::int32_t, 0> a{};
    Uniform<uint, 1> b{};
    Uniform<vec2, 2> c{};
    Uniform<ivec3, 3> d{};
    Uniform<float, 4> e{};
    Uniform<vec4, 5> f{};

    void main() {
        out[0] = vec4(float(a), float(b), c.value.x, c.value.y);
        out[1] = vec4(vec3(d.value), e);
        out[2] = f;
    }
};

// Workgroup built-ins of a 2D dispatch.
struct TC_KERNEL Builtins {
    static constexpr char fileLocation[] = "builtins";
    uvec3 local_size{4, 2, 1};

    BufferBinding<uvec4, 0> out;

    void main() {
        uvec3 g = gl_GlobalInvocationID;
        uint index = g.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x + g.x;
        out[2u * index] = uvec4(gl_LocalInvocationID, gl_LocalInvocationIndex);
        out[2u * index + 1u] = uvec4(gl_WorkGroupID.xy(), gl_NumWorkGroups.xy());
    }
};

// Image formats and host/GPU conversions: unsigned normalized bytes become
// float RGBA, signed integers stay integers.
struct TC_KERNEL ImageFormats {
    static constexpr char fileLocation[] = "image_formats";
    uvec3 local_size{8, 8, 1};

    ImageBinding<gpu::RGBA8, Dim::D2, cpu::RGBA8, 0> colors;
    ImageBinding<gpu::RGBA32F, Dim::D2, cpu::RGB8, 1> floats;
    ImageBinding<gpu::R32I, Dim::D2, cpu::R32I, 2> signedInts;

    void main() {
        ivec2 p = ivec2(gl_GlobalInvocationID.xy());
        ivec2 size = imageSize(colors);
        if (p.x >= size.x or p.y >= size.y) {
            return;
        }
        vec4 c = imageLoad(colors, p);
        imageStore(floats, p, vec4(c.zyx() * 0.5f + vec3{0.25f}, 1.0f));
        imageStore(colors, p, vec4(1.0f) - c);
        ivec4 v = imageLoad(signedInts, p);
        imageStore(signedInts, p, ivec4(v.x * -3 + p.x - p.y));
    }
};

// Buffer atomics: a histogram, order-independent reductions, stream
// compaction (order varies between runs) and exchange / compare-and-swap.
struct TC_KERNEL AtomicOps {
    static constexpr char fileLocation[] = "atomic_ops";
    uvec3 local_size{64, 1, 1};

    BufferBinding<uint, 0> values;
    BufferBinding<uint, 1> bins;
    BufferBinding<uint, 2> stats; // max, min, or, and, xor, compacted count, index sum
    BufferBinding<uint, 3> compacted;
    BufferBinding<std::int32_t, 4> slots;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(values.length())) {
            return;
        }
        uint v = values[i];
        atomicAdd(bins[v % 16u], 1u);
        atomicMax(stats[0], v);
        atomicMin(stats[1], v);
        atomicOr(stats[2], v);
        atomicAnd(stats[3], v);
        atomicXor(stats[4], v);
        if (v % 3u == 0u) {
            uint slot = atomicAdd(stats[5], 1u);
            compacted[slot] = v;
        }
        std::int32_t previous = atomicCompSwap(slots[2u * i], 0, int(v & 0xffffu));
        std::int32_t index = atomicExchange(slots[2u * i + 1u], previous - 5);
        atomicAdd(stats[6], uint(index));
    }
};

// Tree reduction in shared memory, with barrier() inside a loop.
struct TC_KERNEL WorkgroupReduce {
    static constexpr char fileLocation[] = "workgroup_reduce";
    static constexpr bool usesBarrier = true;
    uvec3 local_size{64, 1, 1};

    BufferBinding<uint, 0> values;
    BufferBinding<uint, 1> sums;
    Shared<std::array<uint, 64>> partial;

    void main() {
        uint local = gl_LocalInvocationIndex;
        uint i = gl_GlobalInvocationID.x;
        partial[local] = i < uint(values.length()) ? values[i] : 0u;
        barrier();
        for (uint stride = 32u; stride > 0u; stride >>= 1u) {
            if (local < stride) {
                partial[local] += partial[local + stride];
            }
            barrier();
        }
        if (local == 0u) {
            sums[gl_WorkGroupID.x] = partial[0];
        }
    }
};

// Matrix transpose through a padded 16x16 shared tile (2D workgroups).
struct TC_KERNEL TiledTranspose {
    static constexpr char fileLocation[] = "tiled_transpose";
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

// Per-workgroup histogram in shared memory, merged with buffer atomics.
struct TC_KERNEL SharedHistogram {
    static constexpr char fileLocation[] = "shared_histogram";
    static constexpr bool usesBarrier = true;
    uvec3 local_size{128, 1, 1};

    BufferBinding<uint, 0> values;
    BufferBinding<uint, 1> bins; // 32 bins, then the total count
    Shared<std::array<uint, 32>> counts;
    Shared<uint> total;

    void main() {
        uint local = gl_LocalInvocationIndex;
        if (local < 32u) {
            counts[local] = 0u;
        }
        if (local == 0u) {
            total.value = 0u;
        }
        barrier();
        uint i = gl_GlobalInvocationID.x;
        if (i < uint(values.length())) {
            atomicAdd(counts[values[i] % 32u], 1u);
            atomicAdd(total.value, 1u);
        }
        barrier();
        if (local < 32u) {
            atomicAdd(bins[local], counts[local]);
        }
        if (local == 0u) {
            atomicAdd(bins[32], total.value);
        }
    }
};

// Matrices in buffers, uniforms (push constants on Vulkan) and constants.
struct TC_KERNEL MatrixOps {
    static constexpr char fileLocation[] = "matrix_ops";
    uvec3 local_size{16, 1, 1};

    BufferBinding<mat3, 0> mats;
    BufferBinding<vec4, 1> out;
    Uniform<mat3, 0> colorMatrix{};
    Uniform<mat2x3, 1> projection{};

    static constexpr mat2 kRotate{0.0f, 1.0f, -1.0f, 0.0f};

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(mats.length())) {
            return;
        }
        mat3 m = mats[i];
        vec3 v = vec3{float(i), 1.0f, 2.0f};
        vec3 colored = colorMatrix.value * v;
        vec2 rotated = kRotate * vec2{v.x, v.y};
        vec3 projected = projection.value * rotated;
        out[2u * i] = vec4(colored, determinant(m));
        out[2u * i + 1u] = vec4(projected, (transpose(m) * v).y + m[2][1]);
        mats[i] = inverse(m) * 2.0f;
    }
};

// Packing and bit-manipulation built-ins.
struct TC_KERNEL PackingOps {
    static constexpr char fileLocation[] = "packing_ops";
    uvec3 local_size{32, 1, 1};

    BufferBinding<uint, 0> words;
    BufferBinding<uvec4, 1> out;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        uint w = words[i];
        vec4 unorm = unpackUnorm4x8(w);
        vec4 snorm = unpackSnorm4x8(w);
        vec2 half = unpackHalf2x16(w & 0x3bff3bffu);
        out[2u * i] = uvec4(packUnorm4x8(unorm.wzyx()), packSnorm4x8(snorm), packHalf2x16(half),
                            packUnorm2x16(unorm.xy()));
        out[2u * i + 1u] = uvec4(bitfieldExtract(w, int(i % 24u), 8), uint(bitCount(w)), uint(findMSB(w) + 1),
                                 bitfieldInsert(w, i, 8, 8));
    }
};

// Sampled textures: formats that cannot be storage images on OpenGL ES,
// bilinear filtering with repeat wrapping, and integer textures.
struct TC_KERNEL TextureSampling {
    static constexpr char fileLocation[] = "texture_sampling";
    uvec3 local_size{8, 8, 1};

    TextureBinding<gpu::R8, Dim::D2, cpu::R8, 0> gray;
    TextureBinding<gpu::RGBA8, Dim::D2, cpu::RGBA8, 1, Filter::Linear, Wrap::Repeat> color;
    TextureBinding<gpu::R32UI, Dim::D2, cpu::R32UI, 2> ids;
    BufferBinding<vec4, 3> out;

    void main() {
        ivec2 p = ivec2(gl_GlobalInvocationID.xy());
        ivec2 size = textureSize(gray, 0);
        if (p.x >= size.x or p.y >= size.y) {
            return;
        }
        uint index = uint(p.y * size.x + p.x);
        vec2 uv = (vec2(p) + vec2{0.25f, 0.75f}) / vec2(size) * 1.5f;
        out[index * 3u] = texelFetch(gray, p, 0);
        out[index * 3u + 1u] = texture(color, uv);
        out[index * 3u + 2u] = vec4(texelFetch(ids, p, 0)) + vec4(texture(gray, uv).x);
    }
};

} // namespace tests
