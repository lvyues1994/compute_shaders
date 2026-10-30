// Block-compressed textures: GPU decoders against the CPU backend's.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include <astcenc.h>

#include "equivalence.hpp"
#include "kernels/texture_kernels.hpp"

namespace {

using namespace tc;
using namespace tc::testing_support;

bool isAstc(CompressedFormat const format) {
    return static_cast<std::uint32_t>(format) >= static_cast<std::uint32_t>(CompressedFormat::Astc4x4);
}

// ASTC blocks of a smooth image, from astc-encoder: random bits are mostly
// illegal ASTC encodings.
void encodeAstc(CompressedImage &image) {
    auto const block = compressedFormatInfo(image.format()).block;
    auto config = astcenc_config{};
    ASSERT_EQ(astcenc_config_init(ASTCENC_PRF_LDR, static_cast<unsigned>(block.x), static_cast<unsigned>(block.y), 1,
                                  ASTCENC_PRE_FASTEST, 0, &config),
              ASTCENC_SUCCESS);
    auto *context = static_cast<astcenc_context *>(nullptr);
    ASSERT_EQ(astcenc_context_alloc(&config, 1, &context, nullptr), ASTCENC_SUCCESS);
    for (std::uint32_t level = 0; level < image.levels(); ++level) {
        auto const size = image.extent(level);
        for (std::uint32_t layer = 0; layer < image.layers(); ++layer) {
            auto texels = std::vector<std::uint8_t>(static_cast<std::size_t>(size.x) * size.y * 4);
            for (int y = 0; y < size.y; ++y) {
                for (int x = 0; x < size.x; ++x) {
                    auto *texel = texels.data() + (static_cast<std::size_t>(y) * size.x + x) * 4;
                    texel[0] = static_cast<std::uint8_t>(255 * x / size.x);
                    texel[1] = static_cast<std::uint8_t>((255 * y / size.y + 70 * layer) % 256);
                    texel[2] = static_cast<std::uint8_t>((x * y * 7 + 40 * level) % 256);
                    texel[3] = static_cast<std::uint8_t>(x % 3 == 0 ? 255 : 128);
                }
            }
            void *slices[] = {texels.data()};
            auto source = astcenc_image{static_cast<unsigned>(size.x), static_cast<unsigned>(size.y), 1,
                                        ASTCENC_TYPE_U8, slices};
            auto const swizzle = astcenc_swizzle{ASTCENC_SWZ_R, ASTCENC_SWZ_G, ASTCENC_SWZ_B, ASTCENC_SWZ_A};
            auto *blocks = image.blocks(level).data() + layer * image.layerBytes(level);
            ASSERT_EQ(astcenc_compress_image(context, &source, &swizzle, blocks, image.layerBytes(level), 0),
                      ASTCENC_SUCCESS);
            astcenc_compress_reset(context);
        }
    }
    astcenc_context_free(context);
}

CompressedImage makeImage(CompressedFormat const format, std::uint32_t const layers = 1) {
    auto image = CompressedImage{format, ivec2{20, 12}, 2, layers};
    if (isAstc(format)) {
        encodeAstc(image);
        return image;
    }
    auto seed = 977u + static_cast<std::uint32_t>(format);
    auto const blockBytes = compressedFormatInfo(format).blockBytes;
    for (std::uint32_t level = 0; level < image.levels(); ++level) {
        auto &blocks = image.blocks(level);
        for (std::size_t i = 0; i < blocks.size(); ++i) {
            seed = seed * 1664525u + 1013904223u;
            blocks[i] = static_cast<std::uint8_t>(seed >> 24);
            if (format == CompressedFormat::Bc7 and i % blockBytes == 0 and blocks[i] == 0) {
                blocks[i] = 0x40; // mode 0 of BC7 is a reserved encoding
            }
        }
    }
    return image;
}

std::vector<vec4> sample(Device &device, CompressedImage &image) {
    auto result = std::vector<vec4>{};
    auto kernel = tests::CompressedSampling{};
    kernel.blocks.attach(image);
    kernel.filtered.attach(image);
    device.uploadImage(kernel.blocks);
    device.useKernel(kernel);
    device.bindTexture(kernel.blocks);
    device.bindTexture(kernel.filtered);
    for (std::uint32_t level = 0; level < image.levels(); ++level) {
        auto const size = image.extent(level);
        auto out = BufferResource<vec4>{2 * size.x * size.y};
        kernel.out.attach(out);
        kernel.level = static_cast<int>(level);
        device.uploadBuffer(out);
        device.bindBuffer(kernel.out);
        device.setUniform(kernel.level);
        device.execute(kernel, uvec3{(static_cast<uint>(size.x) + 7u) / 8u, (static_cast<uint>(size.y) + 7u) / 8u, 1u});
        device.downloadBuffer(out);
        result.insert(result.end(), out.begin(), out.end());
    }
    return result;
}

// GPUs interpolate the BC1-BC5 palette with their own rounding (NVIDIA up
// to 6 / 255 from the exact thirds and sevenths).
float toleranceFor(CompressedFormat const format) {
    auto const palette = static_cast<std::uint32_t>(format) <= static_cast<std::uint32_t>(CompressedFormat::Bc5);
    return (palette ? 7.0f : 1.5f) / 255.0f;
}

// The largest difference; filtered lookups (odd entries) may differ by
// another 2 / 255 for the GPU's quantised filter weights.
float largestError(std::vector<vec4> const &expected, std::vector<vec4> const &actual) {
    auto largest = 0.0f;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        for (int c = 0; c < 4; ++c) {
            auto const error = std::abs(expected[i][c] - actual[i][c]);
            largest = std::max(largest, i % 2 == 0 ? error : error - 2.0f / 255.0f);
        }
    }
    return largest;
}

TEST_P(Equivalence, CompressedTextures) {
    auto const supported = gpu->info().limits.compressedFormats;
    auto tested = std::string{};
    for (std::uint32_t f = 0; f < kCompressedFormatCount; ++f) {
        auto const format = static_cast<CompressedFormat>(f);
        auto const name = std::string{compressedFormatInfo(format).name};
        if ((supported & compressedBit(format)) == 0) {
            EXPECT_THROW(
                {
                    auto image = makeImage(format);
                    sample(*gpu, image);
                },
                DeviceError)
                << name;
            continue;
        }
        auto image = makeImage(format);
        auto const expected = sample(*cpu, image);
        auto const actual = sample(*gpu, image);
        ASSERT_EQ(expected.size(), actual.size());
        auto const largest = largestError(expected, actual);
        EXPECT_LE(largest, toleranceFor(format)) << name;
        char measured[32];
        std::snprintf(measured, sizeof measured, "(%.1f)", largest * 255.0f);
        tested += " " + name + measured;
    }
    std::printf("[ info     ] %s '%s' samples:%s\n", gpu->info().backend.c_str(), gpu->info().deviceName.c_str(),
                tested.empty() ? " none" : tested.c_str());
}

// Every level of every layer of a compressed array, attached by `attach`.
std::vector<vec4> sampleArray(Device &device, ivec2 const extent, std::uint32_t const levels, std::uint32_t const layers,
                              std::function<void(tests::CompressedArraySampling &)> const &attach) {
    auto result = std::vector<vec4>{};
    auto kernel = tests::CompressedArraySampling{};
    attach(kernel);
    device.useKernel(kernel);
    device.bindTexture(kernel.blocks);
    device.bindTexture(kernel.filtered);
    for (std::uint32_t level = 0; level < levels; ++level) {
        auto const size = mipExtent(Dim::D2, ivec3{extent, 1}, level);
        auto out = BufferResource<vec4>{2 * size.x * size.y * static_cast<int>(layers)};
        kernel.out.attach(out);
        kernel.level = static_cast<int>(level);
        device.uploadBuffer(out);
        device.bindBuffer(kernel.out);
        device.setUniform(kernel.level);
        device.execute(kernel, uvec3{(static_cast<uint>(size.x) + 7u) / 8u, (static_cast<uint>(size.y) + 7u) / 8u, layers});
        device.downloadBuffer(out);
        result.insert(result.end(), out.begin(), out.end());
    }
    return result;
}

std::vector<vec4> sampleHostArray(Device &device, CompressedImage &image) {
    return sampleArray(device, image.extent(), image.levels(), image.layers(), [&](tests::CompressedArraySampling &k) {
        k.blocks.attach(image);
        k.filtered.attach(image);
        device.uploadImage(k.blocks);
    });
}

// The same blocks streamed into a device image through a staging buffer, in
// pieces: layers 1 and 2 of level 0, then layer 0, then all of level 1.
std::vector<vec4> sampleDeviceArray(Device &device, CompressedImage const &source) {
    auto const layers = source.layers();
    auto image = device.createImage<Dim::D2Array>(source.format(), ivec3{source.extent(), static_cast<int>(layers)},
                                                  source.levels());
    EXPECT_EQ(image.layerBytes(1), source.layerBytes(1));
    auto const level0 = source.layerBytes(0);
    auto const level1 = source.layerBytes(1);
    auto staging = device.createBuffer<std::uint32_t>(static_cast<int>((3 * level0 + 3 * level1) / 4));
    auto *bytes = reinterpret_cast<std::uint8_t *>(staging.data());
    std::memcpy(bytes, source.blocks(0).data() + level0, 2 * level0);
    std::memcpy(bytes + 2 * level0, source.blocks(0).data(), level0);
    std::memcpy(bytes + 3 * level0, source.blocks(1).data(), 3 * level1);
    device.uploadBuffer(staging);
    device.copyBufferToImage(staging, 0, image, 0, 1, 2);
    device.copyBufferToImage(staging, 2 * level0, image, 0, 0, 1);
    device.copyBufferToImage(staging, 3 * level0, image, 1, 0, 3);
    return sampleArray(device, source.extent(), source.levels(), layers, [&](tests::CompressedArraySampling &k) {
        k.blocks.attach(image);
        k.filtered.attach(image);
    });
}

TEST_P(Equivalence, CompressedTextureArrays) {
    auto const limits = gpu->info().limits;
    EXPECT_EQ(limits.compressedArrayFormats & ~limits.compressedFormats, 0u);
    auto tested = std::string{};
    for (std::uint32_t f = 0; f < kCompressedFormatCount; ++f) {
        auto const format = static_cast<CompressedFormat>(f);
        auto const name = std::string{compressedFormatInfo(format).name};
        auto image = makeImage(format, 3);
        if ((limits.compressedArrayFormats & compressedBit(format)) == 0) {
            EXPECT_THROW(gpu->createImage<Dim::D2Array>(format, ivec3{20, 12, 3}, 2), DeviceError) << name;
            EXPECT_THROW(sampleHostArray(*gpu, image), DeviceError) << name;
            continue;
        }
        auto const expected = sampleHostArray(*cpu, image);
        auto const actual = sampleHostArray(*gpu, image);
        ASSERT_EQ(expected.size(), actual.size());
        EXPECT_LE(largestError(expected, actual), toleranceFor(format)) << name;
        auto const layerTexels = static_cast<std::size_t>(2 * 20 * 12);
        EXPECT_NE(expected[2 * 7], expected[layerTexels + 2 * 7]) << name << ": layers differ";
        EXPECT_EQ(sampleDeviceArray(*cpu, image), expected) << name << ": CPU device image";
        EXPECT_EQ(sampleDeviceArray(*gpu, image), actual) << name << ": GPU device image";
        tested += " " + name;
    }
    auto flat = tests::CompressedSampling{};
    auto layered = makeImage(CompressedFormat::Etc2Rgb8, 3);
    flat.blocks.attach(layered);
    EXPECT_THROW(cpu->uploadImage(flat.blocks), DeviceError) << "a 2D binding of an array";
    std::printf("[ info     ] %s '%s' samples arrays:%s\n", gpu->info().backend.c_str(), gpu->info().deviceName.c_str(),
                tested.empty() ? " none" : tested.c_str());
}

TEST(CompressedImages, DecodeKnownBlocks) {
    // BC1, four-colour mode (red > blue): every texel takes (2 red + blue) / 3.
    auto bc1 = CompressedImage{CompressedFormat::Bc1, ivec2{4, 4}};
    std::uint8_t const block[8] = {0x00, 0xF8, 0x1F, 0x00, 0xAA, 0xAA, 0xAA, 0xAA};
    std::memcpy(bc1.blocks(0).data(), block, 8);
    bc1.decode();
    auto const texel = bc1.decoded()->level(0)[ivec3{2, 3, 0}];
    EXPECT_NEAR(texel.channels[0], 170.0f / 255.0f, 1.0f / 255.0f);
    EXPECT_EQ(texel.channels[1], 0.0f);
    EXPECT_NEAR(texel.channels[2], 85.0f / 255.0f, 1.0f / 255.0f);
    EXPECT_EQ(texel.channels[3], 1.0f);

    // An ASTC void-extent block: one colour for the whole block.
    auto astc = CompressedImage{CompressedFormat::Astc8x8, ivec2{10, 3}};
    std::uint16_t const color[4] = {16384, 32768, 49152, 65535};
    for (std::size_t b = 0; b < astc.blocks(0).size(); b += 16) {
        auto *bytes = astc.blocks(0).data() + b;
        std::uint64_t const header = 0xFFFFFFFFFFFFFDFCull; // 2D LDR void extent, no extent coordinates
        std::memcpy(bytes, &header, 8);
        std::memcpy(bytes + 8, color, 8);
    }
    astc.decode();
    auto const constant = astc.decoded()->level(0)[ivec3{9, 2, 0}];
    EXPECT_NEAR(constant.channels[0], 0.25f, 1e-3f);
    EXPECT_NEAR(constant.channels[1], 0.5f, 1e-3f);
    EXPECT_NEAR(constant.channels[2], 0.75f, 1e-3f);
    EXPECT_NEAR(constant.channels[3], 1.0f, 1e-3f);
    EXPECT_EQ(astc.blockCount(0), (ivec2{2, 1}));
    EXPECT_EQ(astc.blocks(0).size(), 32u);
}

} // namespace
