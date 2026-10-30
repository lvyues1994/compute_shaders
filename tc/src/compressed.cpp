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

void decodeBlocks(CompressedImage const &image, std::uint32_t const level, BufferResource<cpu::RGBA32F, Dim::D2> &out) {
    auto const size = image.extent(level);
    auto const blocks = image.blockCount(level);
    auto const bytes = compressedFormatInfo(image.format()).blockBytes;
    auto const &data = image.blocks(level);
    for (int by = 0; by < blocks.y; ++by) {
        for (int bx = 0; bx < blocks.x; ++bx) {
            auto const texels = decodeBlock(image.format(), data.data() + (static_cast<std::size_t>(by) * blocks.x + bx) * bytes);
            for (int y = 0; y < 4 and 4 * by + y < size.y; ++y) {
                for (int x = 0; x < 4 and 4 * bx + x < size.x; ++x) {
                    auto const &texel = texels[static_cast<std::size_t>(4 * y + x)];
                    out[ivec2{4 * bx + x, 4 * by + y}] = cpu::RGBA32F{{texel.x, texel.y, texel.z, texel.w}};
                }
            }
        }
    }
}

void decodeAstc(CompressedImage const &image, std::uint32_t const level, BufferResource<cpu::RGBA32F, Dim::D2> &out) {
    auto const block = compressedFormatInfo(image.format()).block;
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
    auto const size = image.extent(level);
    void *slices[] = {out.data()};
    auto decoded = astcenc_image{};
    decoded.dim_x = static_cast<unsigned>(size.x);
    decoded.dim_y = static_cast<unsigned>(size.y);
    decoded.dim_z = 1;
    decoded.data_type = ASTCENC_TYPE_F32;
    decoded.data = slices;
    auto const swizzle = astcenc_swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
    auto const &data = image.blocks(level);
    status = astcenc_decompress_image(context, data.data(), data.size(), &decoded, &swizzle, 0);
    astcenc_context_free(context);
    if (status != ASTCENC_SUCCESS) {
        throw std::runtime_error{std::string{"astcenc: "} + astcenc_get_error_string(status)};
    }
}

} // namespace

CompressedImage::CompressedImage(CompressedFormat const format, ivec2 const extent, std::uint32_t const levels)
    : formatValue(format), extentValue(extent), id(ResourceIdentity::create()) {
    auto const full = fullMipLevels(Dim::D2, ivec3{extent, 1});
    if (extent.x <= 0 or extent.y <= 0 or levels == 0 or levels > full) {
        throw std::invalid_argument{"CompressedImage: bad extent or level count"};
    }
    data.resize(levels);
    for (std::uint32_t level = 0; level < levels; ++level) {
        auto const blocks = blockCount(level);
        data[level].resize(static_cast<std::size_t>(blocks.x) * blocks.y * compressedFormatInfo(format).blockBytes);
    }
}

CompressedImage::CompressedImage(CompressedImage const &other)
    : formatValue(other.formatValue), extentValue(other.extentValue), data(other.data), id(ResourceIdentity::create()) {}

CompressedImage &CompressedImage::operator=(CompressedImage const &other) {
    if (this != &other) {
        formatValue = other.formatValue;
        extentValue = other.extentValue;
        data = other.data;
        id = ResourceIdentity::create();
        decodedChain.reset();
    }
    return *this;
}

ivec2 CompressedImage::blockCount(std::uint32_t const level) const noexcept {
    auto const size = extent(level);
    auto const block = compressedFormatInfo(formatValue).block;
    return ivec2{(size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y};
}

void CompressedImage::decode() {
    auto chain = std::make_shared<MipChain<cpu::RGBA32F, Dim::D2>>(extentValue, levels());
    for (std::uint32_t level = 0; level < levels(); ++level) {
        if (isAstc(formatValue)) {
            decodeAstc(*this, level, chain->level(level));
        } else {
            decodeBlocks(*this, level, chain->level(level));
        }
    }
    decodedChain = std::move(chain);
}

} // namespace tc
