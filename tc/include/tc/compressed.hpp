#pragma once

#include <cassert>
#include <cstdint>
#include <memory>
#include <vector>

#include <tc/buffer.hpp>
#include <tc/dim.hpp>
#include <tc/mip_chain.hpp>
#include <tc/pixel.hpp>
#include <tc/texture.hpp>
#include <tc/vec.hpp>

namespace tc {

// Block-compressed formats of sampled textures (never storage images).
// Which ones a GPU samples is in DeviceLimits::compressedFormats: BC on
// desktop GPUs, ETC2/EAC everywhere OpenGL ES 3.0 is, ASTC on mobile GPUs.
// The CPU backend decodes all of them.
enum class CompressedFormat : std::uint8_t {
    Bc1,        // RGB + 1-bit alpha, 4x4 blocks of 8 bytes
    Bc3,        // RGBA, 16 bytes
    Bc4,        // R, 8 bytes
    Bc5,        // RG, 16 bytes
    Bc7,        // RGBA, 16 bytes
    Etc2Rgb8,   // RGB, 8 bytes
    Etc2Rgb8A1, // RGB + 1-bit alpha, 8 bytes
    Etc2Rgba8,  // RGBA (EAC alpha), 16 bytes
    EacR11,     // R, 8 bytes
    EacRg11,    // RG, 16 bytes
    Astc4x4,    // RGBA (LDR), 16-byte blocks of 4x4 to 12x12 texels
    Astc5x5,
    Astc6x6,
    Astc8x8,
    Astc10x10,
    Astc12x12,
};

inline constexpr std::uint32_t kCompressedFormatCount = 16;

struct CompressedFormatInfo {
    char const *name;
    ivec2 block;
    std::uint32_t blockBytes;
};

constexpr CompressedFormatInfo compressedFormatInfo(CompressedFormat const format) noexcept {
    constexpr CompressedFormatInfo table[kCompressedFormatCount] = {
        {"bc1", ivec2{4, 4}, 8},          {"bc3", ivec2{4, 4}, 16},         {"bc4", ivec2{4, 4}, 8},
        {"bc5", ivec2{4, 4}, 16},         {"bc7", ivec2{4, 4}, 16},         {"etc2_rgb8", ivec2{4, 4}, 8},
        {"etc2_rgb8a1", ivec2{4, 4}, 8},  {"etc2_rgba8", ivec2{4, 4}, 16},  {"eac_r11", ivec2{4, 4}, 8},
        {"eac_rg11", ivec2{4, 4}, 16},    {"astc_4x4", ivec2{4, 4}, 16},    {"astc_5x5", ivec2{5, 5}, 16},
        {"astc_6x6", ivec2{6, 6}, 16},    {"astc_8x8", ivec2{8, 8}, 16},    {"astc_10x10", ivec2{10, 10}, 16},
        {"astc_12x12", ivec2{12, 12}, 16},
    };
    return table[static_cast<std::uint32_t>(format)];
}

// The DeviceLimits::compressedFormats bit of `format`.
constexpr std::uint32_t compressedBit(CompressedFormat const format) noexcept {
    return 1u << static_cast<std::uint32_t>(format);
}

// A compressed texture with mip levels: the blocks of each level in
// row-major order, as image files (KTX, DDS) store them. A copy is a new
// resource.
struct CompressedImage {
    CompressedImage(CompressedFormat format, ivec2 extent, std::uint32_t levels = 1);
    CompressedImage(CompressedImage const &other);
    CompressedImage &operator=(CompressedImage const &other);
    CompressedImage(CompressedImage &&) noexcept = default;
    CompressedImage &operator=(CompressedImage &&) noexcept = default;
    ~CompressedImage() = default;

    CompressedFormat format() const noexcept { return formatValue; }
    std::uint32_t levels() const noexcept { return static_cast<std::uint32_t>(data.size()); }
    ivec2 extent(std::uint32_t const level = 0) const noexcept {
        auto const size = mipExtent(Dim::D2, ivec3{extentValue, 1}, level);
        return ivec2{size.x, size.y};
    }
    // Blocks across and down level `level`.
    ivec2 blockCount(std::uint32_t level = 0) const noexcept;
    std::vector<std::uint8_t> &blocks(std::uint32_t const level) noexcept { return data[level]; }
    std::vector<std::uint8_t> const &blocks(std::uint32_t const level) const noexcept { return data[level]; }
    ResourceIdentity const &identity() const noexcept { return id; }

    // Decodes every level into RGBA (missing channels as 0, alpha 1), the
    // copy the CPU backend samples; uploadImage() on a CPU device does this.
    void decode();
    MipChain<cpu::RGBA32F, Dim::D2> *decoded() const noexcept { return decodedChain.get(); }

private:
    CompressedFormat formatValue;
    ivec2 extentValue;
    std::vector<std::vector<std::uint8_t>> data;
    ResourceIdentity id;
    std::shared_ptr<MipChain<cpu::RGBA32F, Dim::D2>> decodedChain;
};

// A compressed texture in a kernel, `uniform sampler2D`, sampled like a
// TextureBinding of normalized RGBA.
template <std::uint32_t Binding, Filter F = Filter::Nearest, Wrap W = Wrap::ClampToEdge, Mip M = Mip::None>
struct CompressedTextureBinding {
    static constexpr std::uint32_t binding = Binding;
    static constexpr Filter filter = F;
    static constexpr Wrap wrap = W;
    static constexpr Mip mip = M;

    void attach(CompressedImage &image_) noexcept { image = &image_; }
    CompressedImage *attachedImage() const noexcept { return image; }

private:
    CompressedImage *image = nullptr;
};

namespace detail {

template <std::uint32_t B, Filter F, Wrap W, Mip M>
TextureBinding<gpu::RGBA32F, Dim::D2, cpu::RGBA32F, B, F, W, M>
decodedTexture(CompressedTextureBinding<B, F, W, M> const &texture) noexcept {
    assert(texture.attachedImage() != nullptr and "CompressedTextureBinding used before attach()");
    auto *chain = texture.attachedImage()->decoded();
    assert(chain != nullptr and "upload the compressed texture to the device before executing");
    auto sampler = TextureBinding<gpu::RGBA32F, Dim::D2, cpu::RGBA32F, B, F, W, M>{};
    sampler.attach(*chain);
    return sampler;
}

} // namespace detail

template <std::uint32_t B, Filter F, Wrap W, Mip M>
ivec2 textureSize(CompressedTextureBinding<B, F, W, M> const &texture, std::int32_t const lod) noexcept {
    return textureSize(detail::decodedTexture(texture), lod);
}

template <std::uint32_t B, Filter F, Wrap W, Mip M>
vec4 texelFetch(CompressedTextureBinding<B, F, W, M> const &texture, ivec2 const coord, std::int32_t const lod) noexcept {
    return texelFetch(detail::decodedTexture(texture), coord, lod);
}

template <std::uint32_t B, Filter F, Wrap W, Mip M>
vec4 textureLod(CompressedTextureBinding<B, F, W, M> const &texture, vec2 const &uv, float const lod) noexcept {
    return textureLod(detail::decodedTexture(texture), uv, lod);
}

template <std::uint32_t B, Filter F, Wrap W, Mip M>
vec4 texture(CompressedTextureBinding<B, F, W, M> const &texture, vec2 const &uv) noexcept {
    return textureLod(detail::decodedTexture(texture), uv, 0.0f);
}

} // namespace tc
