#include <tc/compressed.hpp>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

#define BCDEC_IMPLEMENTATION
#define BCDEC_BC4BC5_PRECISE
#include <bcdec.h>
#define ETCDEC_IMPLEMENTATION
#include <etcdec.h>

#include <astcenc.h>

namespace tc {

namespace {

bool isAstc(CompressedFormat const format) noexcept {
    return static_cast<std::uint32_t>(format) >= static_cast<std::uint32_t>(CompressedFormat::Astc4x4);
}

// One 4x4 block of a BC or ETC format as RGBA floats, row-major.
using Texels = std::array<vec4, 16>;

Texels fromRgba8(std::array<std::uint8_t, 64> const &rgba) {
    auto texels = Texels{};
    for (int i = 0; i < 16; ++i) {
        for (int c = 0; c < 4; ++c) {
            texels[static_cast<std::size_t>(i)][c] = static_cast<float>(rgba[static_cast<std::size_t>(4 * i + c)]) / 255.0f;
        }
    }
    return texels;
}

// The first `channels` of each texel from tightly packed floats; the rest
// read as (0, 0, 0, 1).
Texels fromFloats(float const *values, int const channels) {
    auto texels = Texels{};
    for (int i = 0; i < 16; ++i) {
        texels[static_cast<std::size_t>(i)] = vec4{0.0f, 0.0f, 0.0f, 1.0f};
        for (int c = 0; c < channels; ++c) {
            texels[static_cast<std::size_t>(i)][c] = values[channels * i + c];
        }
    }
    return texels;
}

Texels decodeBlock(CompressedFormat const format, std::uint8_t const *block) {
    auto rgba = std::array<std::uint8_t, 64>{};
    auto values = std::array<float, 32>{};
    switch (format) {
    case CompressedFormat::Bc1:
        bcdec_bc1(block, rgba.data(), 16);
        return fromRgba8(rgba);
    case CompressedFormat::Bc3:
        bcdec_bc3(block, rgba.data(), 16);
        return fromRgba8(rgba);
    case CompressedFormat::Bc4:
        bcdec_bc4_float(block, values.data(), 4, 0);
        return fromFloats(values.data(), 1);
    case CompressedFormat::Bc5:
        bcdec_bc5_float(block, values.data(), 8, 0);
        return fromFloats(values.data(), 2);
    case CompressedFormat::Bc7:
        bcdec_bc7(block, rgba.data(), 16);
        return fromRgba8(rgba);
    case CompressedFormat::Etc2Rgb8:
        etcdec_etc_rgb(block, rgba.data(), 16);
        return fromRgba8(rgba);
    case CompressedFormat::Etc2Rgb8A1:
        etcdec_etc_rgb_a1(block, rgba.data(), 16);
        return fromRgba8(rgba);
    case CompressedFormat::Etc2Rgba8:
        etcdec_eac_rgba(block, rgba.data(), 16);
        return fromRgba8(rgba);
    case CompressedFormat::EacR11:
        etcdec_eac_r11_float(block, values.data(), 4 * 4, 0);
        return fromFloats(values.data(), 1);
    case CompressedFormat::EacRg11:
        etcdec_eac_rg11_float(block, values.data(), 8 * 4, 0);
        return fromFloats(values.data(), 2);
    default:
        throw std::logic_error{"not a 4x4 BC or ETC format"};
    }
}

void decodeBlocks(CompressedFormat const format, ivec2 const size, std::uint8_t const *data, cpu::RGBA32F *out) {
    auto const blocks = compressedBlockCount(format, size, 0);
    auto const bytes = compressedFormatInfo(format).blockBytes;
    for (int by = 0; by < blocks.y; ++by) {
        for (int bx = 0; bx < blocks.x; ++bx) {
            auto const texels = decodeBlock(format, data + (static_cast<std::size_t>(by) * blocks.x + bx) * bytes);
            for (int y = 0; y < 4 and 4 * by + y < size.y; ++y) {
                for (int x = 0; x < 4 and 4 * bx + x < size.x; ++x) {
                    auto const &texel = texels[static_cast<std::size_t>(4 * y + x)];
                    out[static_cast<std::size_t>(4 * by + y) * static_cast<std::size_t>(size.x) + 4 * bx + x] =
                        cpu::RGBA32F{{texel.x, texel.y, texel.z, texel.w}};
                }
            }
        }
    }
}

void decodeAstc(CompressedFormat const format, ivec2 const size, std::uint8_t const *data, std::size_t const bytes,
                cpu::RGBA32F *out) {
    auto const block = compressedFormatInfo(format).block;
    auto config = astcenc_config{};
    auto status = astcenc_config_init(ASTCENC_PRF_LDR, static_cast<unsigned>(block.x), static_cast<unsigned>(block.y), 1,
                                      ASTCENC_PRE_FASTEST, ASTCENC_FLG_DECOMPRESS_ONLY, &config);
    auto *context = static_cast<astcenc_context *>(nullptr);
    if (status == ASTCENC_SUCCESS) {
        status = astcenc_context_alloc(&config, 1, &context, nullptr);
    }
    if (status != ASTCENC_SUCCESS) {
        throw std::runtime_error{std::string{"astcenc: "} + astcenc_get_error_string(status)};
    }
    void *slices[] = {out};
    auto decoded = astcenc_image{};
    decoded.dim_x = static_cast<unsigned>(size.x);
    decoded.dim_y = static_cast<unsigned>(size.y);
    decoded.dim_z = 1;
    decoded.data_type = ASTCENC_TYPE_F32;
    decoded.data = slices;
    auto const swizzle = astcenc_swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    status = astcenc_decompress_image(context, data, bytes, &decoded, &swizzle, 0);
    astcenc_context_free(context);
    if (status != ASTCENC_SUCCESS) {
        throw std::runtime_error{std::string{"astcenc: "} + astcenc_get_error_string(status)};
    }
}

} // namespace

ivec2 compressedBlockCount(CompressedFormat const format, ivec2 const extent, std::uint32_t const level) noexcept {
    auto const size = mipExtent(Dim::D2, ivec3{extent, 1}, level);
    auto const block = compressedFormatInfo(format).block;
    return ivec2{(size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y};
}

std::size_t compressedLayerBytes(CompressedFormat const format, ivec2 const extent, std::uint32_t const level) noexcept {
    auto const blocks = compressedBlockCount(format, extent, level);
    return static_cast<std::size_t>(blocks.x) * static_cast<std::size_t>(blocks.y) * compressedFormatInfo(format).blockBytes;
}

void decodeCompressed(CompressedFormat const format, ivec2 const size, std::uint8_t const *blocks,
                      std::size_t const bytes, cpu::RGBA32F *texels) {
    if (bytes != compressedLayerBytes(format, size, 0)) {
        throw std::invalid_argument{"decodeCompressed: the blocks do not cover the image"};
    }
    if (isAstc(format)) {
        decodeAstc(format, size, blocks, bytes, texels);
    } else {
        decodeBlocks(format, size, blocks, texels);
    }
}

CompressedImage::CompressedImage(CompressedFormat const format, ivec2 const extent, std::uint32_t const levels,
                                 std::uint32_t const layers)
    : formatValue(format), extentValue(extent), layerCount(layers), id(ResourceIdentity::create()) {
    auto const full = extent.x > 0 and extent.y > 0 ? fullMipLevels(Dim::D2, ivec3{extent, 1}) : 0u;
    if (levels == 0 or levels > full or layers == 0) {
        throw std::invalid_argument{"CompressedImage: bad extent, level or layer count"};
    }
    data.resize(levels);
    for (std::uint32_t level = 0; level < levels; ++level) {
        data[level].resize(layerBytes(level) * layers);
    }
}

CompressedImage::CompressedImage(CompressedImage const &other)
    : formatValue(other.formatValue), extentValue(other.extentValue), layerCount(other.layerCount), data(other.data),
      id(ResourceIdentity::create()) {}

CompressedImage &CompressedImage::operator=(CompressedImage const &other) {
    if (this != &other) {
        formatValue = other.formatValue;
        extentValue = other.extentValue;
        layerCount = other.layerCount;
        data = other.data;
        id = ResourceIdentity::create();
        decodedChain.reset();
    }
    return *this;
}

void CompressedImage::decode() {
    auto chain = std::make_shared<DecodedChain>(ivec3{extentValue, static_cast<std::int32_t>(layerCount)}, levels());
    for (std::uint32_t level = 0; level < levels(); ++level) {
        auto const size = extent(level);
        auto const bytes = layerBytes(level);
        for (std::uint32_t layer = 0; layer < layerCount; ++layer) {
            auto *texels = chain->level(level).data() + static_cast<std::size_t>(layer) * size.x * size.y;
            decodeCompressed(formatValue, size, data[level].data() + layer * bytes, bytes, texels);
        }
    }
    decodedChain = std::move(chain);
}

} // namespace tc
