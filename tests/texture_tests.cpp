// Mip chains, array textures and 16-bit float formats on every backend.

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "equivalence.hpp"
#include "kernels/texture_kernels.hpp"

namespace {

using namespace tc;
using namespace tc::testing_support;

// Every level a different colour, with gradients to filter.
MipChain<cpu::RGBA8, Dim::D2> gradientChain(ivec2 const size) {
    auto chain = MipChain<cpu::RGBA8, Dim::D2>{size};
    for (std::uint32_t level = 0; level < chain.levels(); ++level) {
        auto &texels = chain.level(level);
        auto const extent = texels.extent();
        for (int y = 0; y < extent.y; ++y) {
            for (int x = 0; x < extent.x; ++x) {
                texels[ivec2{x, y}] = cpu::RGBA8{{static_cast<std::uint8_t>(40 + 50 * level),
                                                  static_cast<std::uint8_t>(255 * x / extent.x),
                                                  static_cast<std::uint8_t>(255 * y / extent.y), 255}};
            }
        }
    }
    return chain;
}

void expectNear(vec4 const &expected, vec4 const &actual, float const tolerance, std::string const &what) {
    for (int c = 0; c < 4; ++c) {
        EXPECT_NEAR(expected[c], actual[c], tolerance) << what << " component " << c;
    }
}

TEST_P(Equivalence, MipSampling) {
    auto const [expected, actual] = runBoth<std::vector<vec4>>([](Device &device) {
        auto chain = gradientChain(ivec2{16, 16});
        auto out = BufferResource<vec4>{8 * 5};
        auto kernel = tests::MipSampling{};
        kernel.flat.attach(chain);
        kernel.nearest.attach(chain);
        kernel.trilinear.attach(chain);
        kernel.out.attach(out);
        device.uploadImage(kernel.trilinear);
        device.uploadBuffer(out);
        device.useKernel(kernel);
        device.bindTexture(kernel.flat);
        device.bindTexture(kernel.nearest);
        device.bindTexture(kernel.trilinear);
        device.bindBuffer(kernel.out);
        device.execute(kernel, uvec3{1u, 1u, 1u});
        device.downloadBuffer(out);
        return contents(out);
    });
    for (std::size_t i = 0; i < expected.size(); ++i) {
        // Filter weights and the level of detail are quantised on GPUs.
        expectNear(expected[i], actual[i], 4.0f / 255.0f, "output " + std::to_string(i));
    }
    auto const levelRed = [](float const level) { return (40.0f + 50.0f * level) / 255.0f; };
    EXPECT_NEAR(actual[2 * 5 + 1].x, levelRed(1.0f), 1e-3f) << "lod 0.75 picks level 1";
    EXPECT_NEAR(actual[4 * 5 + 1].x, levelRed(1.0f), 1e-3f) << "lod 1.4375 picks level 1";
    EXPECT_NEAR(actual[5 * 5 + 2].x, levelRed(2.625f), 2.0f / 255.0f) << "lod 2.625 blends levels 2 and 3";
    EXPECT_NEAR(actual[7 * 5 + 1].x, levelRed(4.0f), 1e-3f) << "lod 9 stops at the last level";
    EXPECT_NEAR(actual[7 * 5 + 0].x, levelRed(0.0f), 1e-3f) << "Mip::None stays on level 0";
    EXPECT_EQ(actual[3 * 5 + 4].x, 8.0f) << "textureSize of level 1";
    EXPECT_EQ(actual[3 * 5 + 4].y, 2.0f) << "textureSize of level 3";
    EXPECT_NEAR(actual[2 * 5 + 3].x, levelRed(2.0f), 1e-3f) << "texelFetch from level 2";
}

std::vector<cpu::RGBA8> flatten(MipChain<cpu::RGBA8, Dim::D2> const &chain) {
    auto texels = std::vector<cpu::RGBA8>{};
    for (std::uint32_t level = 0; level < chain.levels(); ++level) {
        texels.insert(texels.end(), chain.level(level).begin(), chain.level(level).end());
    }
    return texels;
}

// The engine's mip generation: one dispatch per level.
TEST_P(Equivalence, GenerateMips) {
    auto const [expected, actual] = runBoth<std::vector<cpu::RGBA8>>([](Device &device) {
        auto chain = MipChain<cpu::RGBA8, Dim::D2>{ivec2{32, 32}};
        auto &top = chain.level(0);
        for (int y = 0; y < 32; ++y) {
            for (int x = 0; x < 32; ++x) {
                top[ivec2{x, y}] = cpu::RGBA8{{static_cast<std::uint8_t>((x * 37 + y * 11) % 256),
                                               static_cast<std::uint8_t>(x * 8), static_cast<std::uint8_t>(y * 8),
                                               static_cast<std::uint8_t>((x ^ y) * 8)}};
            }
        }
        auto kernel = tests::Downsample{};
        kernel.source.attach(chain);
        device.uploadImage(kernel.source);
        device.useKernel(kernel);
        for (std::uint32_t level = 1; level < chain.levels(); ++level) {
            kernel.target.attach(chain, level);
            kernel.sourceLevel = static_cast<int>(level) - 1;
            device.bindTexture(kernel.source);
            device.bindImage(kernel.target);
            device.setUniform(kernel.sourceLevel);
            auto const groups = static_cast<uint>((chain.extent(level).x + 7) / 8);
            device.execute(kernel, uvec3{groups, groups, 1u});
        }
        device.downloadImage(kernel.source);
        return flatten(chain);
    });
    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        for (int c = 0; c < 4; ++c) {
            EXPECT_NEAR(expected[i].channels[c], actual[i].channels[c], 2) << "texel " << i;
        }
    }
}

// One level of a chain goes up and comes back on its own.
TEST_P(Equivalence, SingleMipLevelTransfers) {
    auto chain = gradientChain(ivec2{16, 16});
    auto texture = tests::Downsample{}.source;
    texture.attach(chain);
    gpu->uploadImage(texture);
    std::fill(chain.level(2).begin(), chain.level(2).end(), cpu::RGBA8{{1, 2, 3, 4}});
    std::fill(chain.level(3).begin(), chain.level(3).end(), cpu::RGBA8{{9, 9, 9, 9}});
    gpu->uploadImage(texture, 2);
    std::fill(chain.level(2).begin(), chain.level(2).end(), cpu::RGBA8{});
    gpu->downloadImage(texture, 2);
    auto const texel = [&](std::uint32_t const level, int const x, int const y) {
        return chain.level(level)[ivec2{x, y}].channels[0 + (level == 2 ? 2 : 0)];
    };
    EXPECT_EQ(texel(2, 3, 1), 3);
    EXPECT_EQ(texel(3, 0, 0), 9) << "level 3 was not downloaded";
    gpu->downloadImage(texture, 3);
    EXPECT_EQ(texel(3, 0, 0), 40 + 50 * 3) << "level 3 was not uploaded";
    EXPECT_THROW(gpu->downloadImage(texture, 5), DeviceError);
}

TEST_P(Equivalence, ArrayTextures) {
    struct Result {
        std::vector<vec4> sampled;
        std::vector<cpu::RGBA8> painted;
    };
    auto const [expected, actual] = runBoth<Result>([](Device &device) {
        auto const size = ivec3{8, 2, 3};
        auto layers = BufferResource<cpu::RGBA8, Dim::D2Array>{size};
        for (int z = 0; z < size.z; ++z) {
            for (int y = 0; y < size.y; ++y) {
                for (int x = 0; x < size.x; ++x) {
                    layers[ivec3{x, y, z}] = cpu::RGBA8{{static_cast<std::uint8_t>(60 * z + 20),
                                                         static_cast<std::uint8_t>(30 * x),
                                                         static_cast<std::uint8_t>(100 * y), 255}};
                }
            }
        }
        auto painted = BufferResource<cpu::RGBA8, Dim::D2Array>{size};
        auto out = BufferResource<vec4>{6 * 3};
        auto kernel = tests::ArrayLayers{};
        kernel.layers.attach(layers);
        kernel.painted.attach(painted);
        kernel.out.attach(out);
        device.uploadImage(kernel.layers);
        device.uploadImage(kernel.painted);
        device.uploadBuffer(out);
        device.useKernel(kernel);
        device.bindTexture(kernel.layers);
        device.bindImage(kernel.painted);
        device.bindBuffer(kernel.out);
        device.execute(kernel, uvec3{1u, 1u, 1u});
        device.downloadBuffer(out);
        device.downloadImage(kernel.painted);
        return Result{contents(out), std::vector<cpu::RGBA8>(painted.begin(), painted.end())};
    });
    for (std::size_t i = 0; i < expected.sampled.size(); ++i) {
        expectNear(expected.sampled[i], actual.sampled[i], 2.0f / 255.0f, "output " + std::to_string(i));
    }
    for (std::size_t i = 0; i < expected.painted.size(); ++i) {
        for (int c = 0; c < 4; ++c) {
            EXPECT_NEAR(expected.painted[i].channels[c], actual.painted[i].channels[c], 1) << "texel " << i;
        }
    }
    auto const layerRed = [](int const layer) { return (60.0f * static_cast<float>(layer) + 20.0f) / 255.0f; };
    EXPECT_NEAR(actual.sampled[1 * 3].x, layerRed(1), 1e-3f) << "layer 1.25 reads layer 1";
    EXPECT_NEAR(actual.sampled[2 * 3].x, layerRed(2), 1e-3f) << "layer 1.75 reads layer 2";
    EXPECT_NEAR(actual.sampled[4 * 3].x, layerRed(0), 1e-3f) << "layer -1 clamps to 0";
    EXPECT_NEAR(actual.sampled[5 * 3].x, layerRed(2), 1e-3f) << "layer 7 clamps to 2";
    EXPECT_EQ(actual.sampled[2], (vec4{8.0f, 2.0f, 3.0f, 0.0f}));
    // invocation 4 painted (4, 1) of layer 1.
    EXPECT_NEAR(actual.painted[(1 * 2 + 1) * 8 + 4].channels[1], 128, 1) << "0.5 in unorm8";
}

TEST_P(Equivalence, HalfFloatImages) {
    auto const size = ivec2{24, 16};
    auto const [expected, actual] = runBoth<std::vector<cpu::RGBA32F>>([size](Device &device) {
        auto radiance = BufferResource<cpu::RGBA16F, Dim::D2>{size};
        for (int y = 0; y < size.y; ++y) {
            for (int x = 0; x < size.x; ++x) {
                auto const value = static_cast<float>(x * x * 3 + y);
                radiance[ivec2{x, y}] =
                    cpu::RGBA16F{{half{value}, half{0.001f * value}, half{2000.0f - value}, half{1.0f}}};
            }
        }
        auto mapped = BufferResource<cpu::RGBA32F, Dim::D2>{size};
        auto kernel = tests::HalfFloats{};
        kernel.radiance.attach(radiance);
        kernel.mapped.attach(mapped);
        device.uploadImage(kernel.radiance);
        device.uploadImage(kernel.mapped);
        device.useKernel(kernel);
        device.bindTexture(kernel.radiance);
        device.bindImage(kernel.mapped);
        device.execute(kernel, uvec3{3u, 2u, 1u});
        device.downloadImage(kernel.mapped);
        return std::vector<cpu::RGBA32F>(mapped.begin(), mapped.end());
    });
    for (std::size_t i = 0; i < expected.size(); ++i) {
        for (int c = 0; c < 4; ++c) {
            auto const e = expected[i].channels[c];
            // Stored as half: about 3 significant decimal digits.
            EXPECT_NEAR(e, actual[i].channels[c], 2e-3f * std::abs(e) + 1e-3f) << "texel " << i << " channel " << c;
        }
    }
    EXPECT_NEAR(actual[0].channels[3], 500.0f, 0.5f);
    EXPECT_NE(actual[5].channels[0], 0.0f);
}

TEST(HalfFloat, ConvertsLikeTheGpu) {
    EXPECT_EQ(half{1.0f}.bits, 0x3c00u);
    EXPECT_EQ(half{-2.0f}.bits, 0xc000u);
    EXPECT_EQ(half{65504.0f}.bits, 0x7bffu);
    EXPECT_EQ(half{1e6f}.bits, 0x7c00u) << "overflow to infinity";
    EXPECT_EQ(static_cast<float>(half{0.1f}), 0.0999755859375f);
    auto const pixel = convertPixel<cpu::RGBA8>(cpu::RGBA16F{{half{0.5f}, half{2.0f}, half{-1.0f}, half{1.0f}}});
    EXPECT_EQ(pixel.channels[0], 128);
    EXPECT_EQ(pixel.channels[1], 255);
    EXPECT_EQ(pixel.channels[2], 0);
    EXPECT_EQ(cpu::RGBA16F{}.get<Channel::A>().bits, 0u);
    EXPECT_EQ(cpu::R16F{}.get<Channel::A>().bits, 0x3c00u) << "missing alpha reads as 1";
}

TEST(MipChain, HalvesEveryLevel) {
    auto const chain = MipChain<cpu::RGBA8, Dim::D2>{ivec2{20, 6}};
    ASSERT_EQ(chain.levels(), 5u);
    EXPECT_EQ(chain.extent(1), (ivec2{10, 3}));
    EXPECT_EQ(chain.extent(2), (ivec2{5, 1}));
    EXPECT_EQ(chain.extent(4), (ivec2{1, 1}));
    auto const layers = MipChain<cpu::RGBA8, Dim::D2Array>{ivec3{8, 8, 6}, 3};
    EXPECT_EQ(layers.extent(2), (ivec3{2, 2, 6})) << "array layers stay";
    auto const copy = chain;
    EXPECT_NE(copy.identity().key(), chain.identity().key());
}

} // namespace
