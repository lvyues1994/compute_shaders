#pragma once

#include <array>
#include <cstdint>

#include <tc/tc.hpp>

// Kernels for mip chains, array textures and 16-bit float images.
namespace tests {

using namespace tc;

// One chain behind three samplers: no mips, nearest level, blended levels.
struct TC_KERNEL MipSampling {
    static constexpr char fileLocation[] = "mip_sampling";
    uvec3 local_size{8, 1, 1};

    TextureBinding<gpu::RGBA8, Dim::D2, cpu::RGBA8, 0, Filter::Linear, Wrap::ClampToEdge, Mip::None> flat;
    TextureBinding<gpu::RGBA8, Dim::D2, cpu::RGBA8, 1, Filter::Linear, Wrap::ClampToEdge, Mip::Nearest> nearest;
    TextureBinding<gpu::RGBA8, Dim::D2, cpu::RGBA8, 2, Filter::Linear, Wrap::Repeat, Mip::Linear> trilinear;
    BufferBinding<vec4, 3> out;

    // No ties between two levels, where GPUs may round either way.
    static constexpr std::array<float, 8> kLods{0.0f, 0.25f, 0.75f, 1.25f, 1.4375f, 2.625f, 3.75f, 9.0f};

    // texelFetch and textureSize use the same level in every invocation:
    // llvmpipe reads level 0 when the level differs between invocations.
    void main() {
        uint i = gl_LocalInvocationIndex;
        float lod = kLods[i];
        vec2 uv = vec2(0.3f + 0.05f * float(i), 0.6f);
        ivec2 size1 = textureSize(trilinear, 1);
        ivec2 size3 = textureSize(trilinear, 3);
        out[i * 5u] = textureLod(flat, uv, lod);
        out[i * 5u + 1u] = textureLod(nearest, uv, lod);
        out[i * 5u + 2u] = textureLod(trilinear, uv, lod);
        out[i * 5u + 3u] = texelFetch(nearest, ivec2(int(i % 4u), 3), 2);
        out[i * 5u + 4u] = vec4(float(size1.x), float(size3.x), texture(trilinear, uv).x, float(size1.y));
    }
};

// Generates level n of a chain from level n - 1: the chain is sampled as a
// texture and written as an image of one level.
struct TC_KERNEL Downsample {
    static constexpr char fileLocation[] = "downsample";
    uvec3 local_size{8, 8, 1};

    TextureBinding<gpu::RGBA8, Dim::D2, cpu::RGBA8, 0, Filter::Nearest, Wrap::ClampToEdge, Mip::Nearest> source;
    ImageBinding<gpu::RGBA8, Dim::D2, cpu::RGBA8, 1> target;
    Uniform<int, 0> sourceLevel{};

    void main() {
        ivec2 p = ivec2(gl_GlobalInvocationID.xy());
        ivec2 size = imageSize(target);
        if (p.x >= size.x or p.y >= size.y) {
            return;
        }
        ivec2 s = p * 2;
        int level = sourceLevel.value;
        vec4 sum = texelFetch(source, s, level) + texelFetch(source, s + ivec2(1, 0), level)
                 + texelFetch(source, s + ivec2(0, 1), level) + texelFetch(source, s + ivec2(1, 1), level);
        imageStore(target, p, sum * 0.25f);
    }
};

struct TC_KERNEL ArrayLayers {
    static constexpr char fileLocation[] = "array_layers";
    uvec3 local_size{6, 1, 1};

    TextureBinding<gpu::RGBA8, Dim::D2Array, cpu::RGBA8, 0, Filter::Linear, Wrap::Repeat> layers;
    ImageBinding<gpu::RGBA8, Dim::D2Array, cpu::RGBA8, 1> painted;
    BufferBinding<vec4, 2> out;

    // Layer coordinates round to the nearest layer and clamp.
    static constexpr std::array<float, 6> kLayers{0.0f, 1.25f, 1.75f, 2.4f, -1.0f, 7.0f};

    void main() {
        uint i = gl_LocalInvocationIndex;
        int layer = int(i % 3u);
        ivec3 size = textureSize(layers, 0);
        out[i * 3u] = texture(layers, vec3(0.2f + 0.13f * float(i), 0.7f, kLayers[i]));
        out[i * 3u + 1u] = texelFetch(layers, ivec3(int(i), 1, layer), 0);
        out[i * 3u + 2u] = vec4(float(size.x), float(size.y), float(size.z), 0.0f);
        imageStore(painted, ivec3(int(i), 1, layer), vec4(float(i) / 8.0f, 0.5f, 1.0f, 1.0f));
    }
};

// A filtered 16-bit float texture and a 16-bit float storage image whose
// host copy is 32-bit float.
struct TC_KERNEL HalfFloats {
    static constexpr char fileLocation[] = "half_floats";
    uvec3 local_size{8, 8, 1};

    TextureBinding<gpu::RGBA16F, Dim::D2, cpu::RGBA16F, 0, Filter::Linear> radiance;
    ImageBinding<gpu::RGBA16F, Dim::D2, cpu::RGBA32F, 1> mapped;

    void main() {
        ivec2 p = ivec2(gl_GlobalInvocationID.xy());
        ivec2 size = imageSize(mapped);
        if (p.x >= size.x or p.y >= size.y) {
            return;
        }
        vec2 uv = (vec2(p) + vec2(0.25f)) / vec2(size);
        vec4 c = texture(radiance, uv);
        imageStore(mapped, p, c / (c + vec4(1.0f)) * 1000.0f);
    }
};

// Every texel of one level of a compressed texture, and a filtered lookup.
struct TC_KERNEL CompressedSampling {
    static constexpr char fileLocation[] = "compressed_sampling";
    uvec3 local_size{8, 8, 1};

    CompressedTextureBinding<0, Filter::Nearest, Wrap::ClampToEdge, Mip::Nearest> blocks;
    CompressedTextureBinding<1, Filter::Linear, Wrap::Repeat> filtered;
    BufferBinding<vec4, 2> out;
    Uniform<int, 0> level{};

    void main() {
        ivec2 p = ivec2(gl_GlobalInvocationID.xy());
        ivec2 size = textureSize(blocks, level.value);
        if (p.x >= size.x or p.y >= size.y) {
            return;
        }
        uint i = uint(p.y * size.x + p.x);
        out[i * 2u] = texelFetch(blocks, p, level.value);
        out[i * 2u + 1u] = texture(filtered, (vec2(p) + vec2(0.25f)) / vec2(size));
    }
};

} // namespace tests
