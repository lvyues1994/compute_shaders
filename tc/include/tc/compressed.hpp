#pragma once

#include <cassert>
#include <cstddef>
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

// Blocks across and down mip level `level` of an image `extent` texels large.
ivec2 compressedBlockCount(CompressedFormat format, ivec2 extent, std::uint32_t level) noexcept;
// Bytes of one layer of mip level `level`.
std::size_t compressedLayerBytes(CompressedFormat format, ivec2 extent, std::uint32_t level) noexcept;
// Decodes one layer of one level, `size` texels: blocks in row-major order
// to RGBA texels in row-major order (missing channels as 0, alpha as 1).
void decodeCompressed(CompressedFormat format, ivec2 size, std::uint8_t const *blocks, std::size_t bytes,
                      cpu::RGBA32F *texels);

// What the CPU backend samples: every level of every layer, decoded.
using DecodedChain = MipChain<cpu::RGBA32F, Dim::D2Array>;

// A compressed texture with mip levels and array layers: for each level,
// the blocks of layer 0 in row-major order, then those of layer 1, and so
// on, as KTX2 files store them. A copy is a new resource.
struct CompressedImage {
    CompressedImage(CompressedFormat format, ivec2 extent, std::uint32_t levels = 1, std::uint32_t layers = 1);
    CompressedImage(CompressedImage const &other);
    CompressedImage &operator=(CompressedImage const &other);
    CompressedImage(CompressedImage &&) noexcept = default;
    CompressedImage &operator=(CompressedImage &&) noexcept = default;
    ~CompressedImage() = default;

    CompressedFormat format() const noexcept { return formatValue; }
    std::uint32_t levels() const noexcept { return static_cast<std::uint32_t>(data.size()); }
    std::uint32_t layers() const noexcept { return layerCount; }
    ivec2 extent(std::uint32_t const level = 0) const noexcept {
        auto const size = mipExtent(Dim::D2, ivec3{extentValue, 1}, level);
        return ivec2{size.x, size.y};
    }
    // Blocks across and down level `level`.
    ivec2 blockCount(std::uint32_t const level = 0) const noexcept {
        return compressedBlockCount(formatValue, extentValue, level);
    }
    // Bytes of one layer of `level`; blocks(level) holds layers() of them.
    std::size_t layerBytes(std::uint32_t const level = 0) const noexcept {
        return compressedLayerBytes(formatValue, extentValue, level);
    }
    std::vector<std::uint8_t> &blocks(std::uint32_t const level) noexcept { return data[level]; }
    std::vector<std::uint8_t> const &blocks(std::uint32_t const level) const noexcept { return data[level]; }
    ResourceIdentity const &identity() const noexcept { return id; }

    // Decodes every level of every layer into RGBA, the copy the CPU
    // backend samples; uploadImage() on a CPU device does this.
    void decode();
    DecodedChain *decoded() const noexcept { return decodedChain.get(); }

private:
    CompressedFormat formatValue;
    ivec2 extentValue;
    std::uint32_t layerCount;
    std::vector<std::vector<std::uint8_t>> data;
    ResourceIdentity id;
    std::shared_ptr<DecodedChain> decodedChain;
};

namespace detail {

// What the copies of one DeviceCompressedImage share.
struct DeviceCompressedState {
    DeviceCompressedState(CompressedFormat const format_, ivec2 const extent_, std::uint32_t const levels_,
                          std::uint32_t const layers_)
        : format(format_), extent(extent_), levels(levels_), layers(layers_), identity(ResourceIdentity::create()) {}

    CompressedFormat format;
    ivec2 extent;
    std::uint32_t levels;
    std::uint32_t layers;
    ResourceIdentity identity;
    // The decoded image on the CPU backend; empty on GPU devices.
    std::unique_ptr<DecodedChain> host;
};

} // namespace detail

// A compressed texture that lives only on the device that created it
// (Device::createImage with a CompressedFormat), the compressed counterpart
// of DeviceImage: no host copy of its blocks. Device::copyBufferToImage
// fills layers of one level from blocks in a buffer, e.g. a staging ring an
// engine streams textures through; kernels sample it through a
// CompressedTextureBinding of the same Dim. The contents start undefined.
// Copies refer to the same image. D is Dim::D2 or Dim::D2Array.
template <Dim D>
struct DeviceCompressedImage {
    static_assert(D == Dim::D2 or D == Dim::D2Array, "compressed images are 2D or 2D arrays");

    using IndexType = typename DimTraits<D>::IndexType;
    static constexpr Dim dim = D;

    DeviceCompressedImage() = default; // no image

    explicit operator bool() const noexcept { return state != nullptr; }
    CompressedFormat format() const noexcept { return state->format; }
    std::uint32_t levels() const noexcept { return state->levels; }
    std::uint32_t layers() const noexcept { return state->layers; }
    // Array textures: layers in z.
    IndexType extent(std::uint32_t const level = 0) const noexcept {
        auto const size = mipExtent(Dim::D2, ivec3{state->extent, 1}, level);
        if constexpr (D == Dim::D2) {
            return ivec2{size.x, size.y};
        } else {
            return ivec3{size.x, size.y, static_cast<std::int32_t>(state->layers)};
        }
    }
    std::size_t layerBytes(std::uint32_t const level = 0) const noexcept {
        return compressedLayerBytes(state->format, state->extent, level);
    }
    ResourceIdentity const &identity() const noexcept { return state->identity; }

    detail::DeviceCompressedState const *shared() const noexcept { return state.get(); }

private:
    friend struct Device;
    explicit DeviceCompressedImage(std::shared_ptr<detail::DeviceCompressedState> state_) noexcept
        : state(std::move(state_)) {}

    std::shared_ptr<detail::DeviceCompressedState> state;
};

// A compressed texture in a kernel, `uniform sampler2D` (or
// `sampler2DArray` for Dim::D2Array), sampled like a TextureBinding of
// normalized RGBA. Attach a CompressedImage or a DeviceCompressedImage.
template <std::uint32_t Binding, Filter F = Filter::Nearest, Wrap W = Wrap::ClampToEdge, Mip M = Mip::None,
          Dim D = Dim::D2>
struct CompressedTextureBinding {
    static_assert(D == Dim::D2 or D == Dim::D2Array, "compressed textures are 2D or 2D arrays");

    static constexpr std::uint32_t binding = Binding;
    static constexpr Filter filter = F;
    static constexpr Wrap wrap = W;
    static constexpr Mip mip = M;
    static constexpr Dim dim = D;

    void attach(CompressedImage &image_) noexcept {
        image = &image_;
        device = nullptr;
    }
    void attach(DeviceCompressedImage<D> const &image_) noexcept {
        assert(image_);
        image = nullptr;
        device = image_.shared();
    }
    CompressedImage *attachedImage() const noexcept { return image; }
    detail::DeviceCompressedState const *attachedDeviceImage() const noexcept { return device; }
    // What the CPU backend samples.
    DecodedChain *decoded() const noexcept {
        return image != nullptr ? image->decoded() : device != nullptr ? device->host.get() : nullptr;
    }

private:
    CompressedImage *image = nullptr;
    detail::DeviceCompressedState const *device = nullptr;
};

namespace detail {

// 2D textures sample layer 0 of the decoded array.
template <std::uint32_t B, Filter F, Wrap W, Mip M, Dim D>
TextureBinding<gpu::RGBA32F, Dim::D2Array, cpu::RGBA32F, B, F, W, M>
decodedTexture(CompressedTextureBinding<B, F, W, M, D> const &texture) noexcept {
    auto *chain = texture.decoded();
    assert(chain != nullptr and "attach and upload the compressed texture before executing");
    auto sampler = TextureBinding<gpu::RGBA32F, Dim::D2Array, cpu::RGBA32F, B, F, W, M>{};
    sampler.attach(*chain);
    return sampler;
}

} // namespace detail

template <std::uint32_t B, Filter F, Wrap W, Mip M, Dim D>
typename DimTraits<D>::IndexType textureSize(CompressedTextureBinding<B, F, W, M, D> const &texture,
                                             std::int32_t const lod) noexcept {
    auto const size = textureSize(detail::decodedTexture(texture), lod);
    if constexpr (D == Dim::D2) {
        return ivec2{size.x, size.y};
    } else {
        return size;
    }
}

template <std::uint32_t B, Filter F, Wrap W, Mip M, Dim D>
vec4 texelFetch(CompressedTextureBinding<B, F, W, M, D> const &texture, typename DimTraits<D>::IndexType const coord,
                std::int32_t const lod) noexcept {
    if constexpr (D == Dim::D2) {
        return texelFetch(detail::decodedTexture(texture), ivec3{coord.x, coord.y, 0}, lod);
    } else {
        return texelFetch(detail::decodedTexture(texture), coord, lod);
    }
}

template <std::uint32_t B, Filter F, Wrap W, Mip M, Dim D>
vec4 textureLod(CompressedTextureBinding<B, F, W, M, D> const &texture, typename detail::TexCoord<D>::type const &uv,
                float const lod) noexcept {
    if constexpr (D == Dim::D2) {
        return textureLod(detail::decodedTexture(texture), vec3{uv.x, uv.y, 0.0f}, lod);
    } else {
        return textureLod(detail::decodedTexture(texture), uv, lod);
    }
}

template <std::uint32_t B, Filter F, Wrap W, Mip M, Dim D>
vec4 texture(CompressedTextureBinding<B, F, W, M, D> const &texture, typename detail::TexCoord<D>::type const &uv) noexcept {
    return textureLod(texture, uv, 0.0f);
}

} // namespace tc
