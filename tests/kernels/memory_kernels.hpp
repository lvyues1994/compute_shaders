#pragma once

#include <tc/tc.hpp>

// Kernels for device-only buffers, heaps and aliasing.
namespace tests {

using namespace tc;

struct TC_KERNEL FillWords {
    static constexpr char fileLocation[] = "fill_words";
    uvec3 local_size{64, 1, 1};

    BufferBinding<uint, 0> out;
    Uniform<uint, 0> seed{};

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i < uint(out.length())) {
            out[i] = i * seed.value + 1u;
        }
    }
};

// out[i] = in[i] + addend, after reading `in` `rounds` more times (into
// `sink`): a slow reader, so that a missing barrier lets a later write
// overtake it.
struct TC_KERNEL SlowCopy {
    static constexpr char fileLocation[] = "slow_copy";
    uvec3 local_size{64, 1, 1};

    BufferBinding<uint, 0> in;
    BufferBinding<uint, 1> out;
    BufferBinding<uint, 2> sink;
    Uniform<uint, 0> addend{};
    Uniform<uint, 1> rounds{};

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(in.length())) {
            return;
        }
        uint value = 0u;
        for (uint r = 0u; r < rounds.value; ++r) {
            value = (value * 2654435761u) ^ in[(i + r) % uint(in.length())];
        }
        sink[i] = value;
        out[i] = in[i] + addend.value;
    }
};

struct TC_KERNEL FillColour {
    static constexpr char fileLocation[] = "fill_colour";
    uvec3 local_size{8, 8, 1};

    ImageBinding<gpu::RGBA16F, Dim::D2, cpu::RGBA32F, 0> image;
    Uniform<float, 0> seed{};

    void main() {
        ivec2 p = ivec2(gl_GlobalInvocationID.xy());
        ivec2 size = imageSize(image);
        if (p.x < size.x and p.y < size.y) {
            imageStore(image, p, vec4(float(p.x), float(p.y), seed.value, 1.0f));
        }
    }
};

struct TC_KERNEL ReadColour {
    static constexpr char fileLocation[] = "read_colour";
    uvec3 local_size{8, 8, 1};

    TextureBinding<gpu::RGBA16F, Dim::D2, cpu::RGBA32F, 0> image;
    BufferBinding<vec4, 1> out;

    void main() {
        ivec2 p = ivec2(gl_GlobalInvocationID.xy());
        ivec2 size = textureSize(image, 0);
        if (p.x < size.x and p.y < size.y) {
            out[uint(p.y * size.x + p.x)] = texelFetch(image, p, 0);
        }
    }
};

} // namespace tests
