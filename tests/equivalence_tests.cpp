// Runs the same kernels on the CPU backend and on every GPU backend that can
// be created on this machine, and checks that the results agree.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <tc/cpu/cpu_device.hpp>
#include <tc/devices.hpp>

#include <game_of_life/simulate.hpp>
#include <nv12_to_rgba/convert.hpp>
#include <raytracer/render.hpp>

#include "kernels/feature_kernels.hpp"

namespace {

using namespace tc;

struct BackendCase {
    std::string backend;
    std::string deviceNameContains;
};

std::string caseName(testing::TestParamInfo<BackendCase> const &info) {
    auto name = info.param.backend + (info.param.deviceNameContains.empty() ? "" : "_" + info.param.deviceNameContains);
    for (auto &c : name) {
        c = std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
    }
    return name;
}

std::vector<BackendCase> gpuCases() {
    auto cases = std::vector<BackendCase>{};
    for (auto const &backend : availableBackends()) {
        if (backend == "cpu") {
            continue;
        }
        cases.push_back(BackendCase{backend, ""});
        cases.push_back(BackendCase{backend, "llvmpipe"});
    }
    return cases;
}

struct Equivalence : testing::TestWithParam<BackendCase> {
    void SetUp() override {
        auto request = DeviceRequest{};
        request.backend = GetParam().backend;
        request.deviceNameContains = GetParam().deviceNameContains;
        try {
            gpu = createDevice(request);
        } catch (DeviceError const &error) {
            GTEST_SKIP() << "backend unavailable: " << error.what();
        }
        cpu = cpu::createDevice();
    }

    // Runs `body` on both devices; skips when the GPU backend reports that
    // the kernel cannot run there (e.g. features OpenGL ES lacks).
    template <class Result>
    std::pair<Result, Result> runBoth(std::function<Result(Device &)> const &body) {
        auto expected = body(*cpu);
        try {
            return {expected, body(*gpu)};
        } catch (DeviceError const &error) {
            if (std::string{error.what()}.find("cannot run on OpenGL ES") != std::string::npos) {
                throw SkipKernel{error.what()};
            }
            throw;
        }
    }

    struct SkipKernel {
        std::string reason;
    };

    std::unique_ptr<Device> gpu;
    std::unique_ptr<Device> cpu;
};

template <class T>
std::vector<T> contents(BufferResource<T> const &buffer) {
    return std::vector<T>(buffer.begin(), buffer.end());
}

TEST_P(Equivalence, ConstStructArray) {
    auto const [expected, actual] = runBoth<std::vector<vec4>>([](Device &device) {
        auto out = BufferResource<vec4>{8};
        auto kernel = tests::ConstStructArray{};
        kernel.out.attach(out);
        device.uploadBuffer(out);
        device.useKernel(kernel);
        device.bindBuffer(kernel.out);
        device.execute(kernel, uvec3{1u, 1u, 1u});
        device.downloadBuffer(out);
        return contents(out);
    });
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i], actual[i]) << "element " << i << ": got " << actual[i].x << " " << actual[i].y << " "
                                          << actual[i].z << " " << actual[i].w;
    }
}

TEST_P(Equivalence, MathFunctions) {
    auto const [expected, actual] = runBoth<std::vector<vec4>>([](Device &device) {
        constexpr int n = 256;
        auto input = BufferResource<float>{n};
        for (int i = 0; i < n; ++i) {
            input[i] = static_cast<float>(i) / n;
        }
        auto output = BufferResource<vec4>{4 * n};
        auto kernel = tests::MathFunctions{};
        kernel.input.attach(input);
        kernel.output.attach(output);
        device.uploadBuffer(input);
        device.uploadBuffer(output);
        device.useKernel(kernel);
        device.bindBuffer(kernel.input);
        device.bindBuffer(kernel.output);
        device.execute(kernel, uvec3{n / 64, 1u, 1u});
        device.downloadBuffer(output);
        return contents(output);
    });
    for (std::size_t i = 0; i < expected.size(); ++i) {
        for (int c = 0; c < 4; ++c) {
            EXPECT_NEAR(expected[i][c], actual[i][c], 2e-3f) << "element " << i << " component " << c;
        }
    }
}

TEST_P(Equivalence, IntegerOps) {
    struct Result {
        std::vector<std::int32_t> ints;
        std::vector<uint> uints;
    };
    auto const [expected, actual] = runBoth<Result>([](Device &device) {
        constexpr int n = 64;
        auto ints = BufferResource<std::int32_t>{3 * n};
        auto uints = BufferResource<uint>{2 * n};
        auto kernel = tests::IntegerOps{};
        kernel.ints.attach(ints);
        kernel.uints.attach(uints);
        device.uploadBuffer(ints);
        device.uploadBuffer(uints);
        device.useKernel(kernel);
        device.bindBuffer(kernel.ints);
        device.bindBuffer(kernel.uints);
        device.execute(kernel, uvec3{n / 32, 1u, 1u});
        device.downloadBuffer(ints);
        device.downloadBuffer(uints);
        return Result{contents(ints), contents(uints)};
    });
    EXPECT_EQ(expected.ints, actual.ints);
    EXPECT_EQ(expected.uints, actual.uints);
}

TEST_P(Equivalence, ParticleStep) {
    auto const [expected, actual] = runBoth<std::vector<tests::Particle>>([](Device &device) {
        constexpr int n = 100;
        auto particles = BufferResource<tests::Particle>{n};
        for (int i = 0; i < n; ++i) {
            auto &p = particles[i];
            p.position = vec4{static_cast<float>(i), 0.0f, 1.0f, 1.0f};
            p.velocity = vec4{0.0f, 1.0f, static_cast<float>(i % 7), 0.0f};
            p.mass = 1.0f + static_cast<float>(i % 3);
            p.id = i;
        }
        auto kernel = tests::ParticleStep{};
        kernel.dt = 0.5f;
        kernel.gravity = vec3{0.0f, -9.81f, 0.0f};
        kernel.particles.attach(particles);
        device.uploadBuffer(particles);
        device.useKernel(kernel);
        device.bindBuffer(kernel.particles);
        device.setUniform(kernel.dt);
        device.setUniform(kernel.gravity);
        device.execute(kernel, uvec3{2u, 1u, 1u});
        device.downloadBuffer(particles);
        return contents(particles);
    });
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i].id, actual[i].id);
        EXPECT_FLOAT_EQ(expected[i].mass, actual[i].mass);
        for (int c = 0; c < 4; ++c) {
            EXPECT_NEAR(expected[i].position[c], actual[i].position[c], 1e-4f) << "particle " << i;
            EXPECT_NEAR(expected[i].velocity[c], actual[i].velocity[c], 1e-4f) << "particle " << i;
        }
    }
}

// Buffers from createBuffer(): CPU writes and GPU results share memory, and
// upload/download only synchronise.
TEST_P(Equivalence, ZeroCopyBuffers) {
    auto const [expected, actual] = runBoth<std::vector<tests::Particle>>([](Device &device) {
        constexpr int n = 100;
        auto particles = device.createBuffer<tests::Particle>(n);
        for (int i = 0; i < n; ++i) {
            auto &p = particles[i];
            p.position = vec4{static_cast<float>(i), 2.0f, 0.0f, 1.0f};
            p.velocity = vec4{1.0f, 0.0f, static_cast<float>(i % 5), 0.0f};
            p.mass = 2.0f;
            p.id = 7 * i;
        }
        auto kernel = tests::ParticleStep{};
        kernel.dt = 0.25f;
        kernel.gravity = vec3{0.0f, -1.0f, 0.5f};
        kernel.particles.attach(particles);
        device.uploadBuffer(particles);
        device.useKernel(kernel);
        device.bindBuffer(kernel.particles);
        device.setUniform(kernel.dt);
        device.setUniform(kernel.gravity);
        device.execute(kernel, uvec3{2u, 1u, 1u});
        device.execute(kernel, uvec3{2u, 1u, 1u});
        device.downloadBuffer(particles);
        return contents(particles);
    });
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(expected[i].id, actual[i].id);
        for (int c = 0; c < 4; ++c) {
            EXPECT_NEAR(expected[i].position[c], actual[i].position[c], 1e-4f) << "particle " << i;
        }
    }
    if (GetParam().backend == "vulkan") {
        EXPECT_TRUE(gpu->createBuffer<float>(16).usesExternalMemory());
    }
}

TEST_P(Equivalence, UniformTypes) {
    auto const [expected, actual] = runBoth<std::vector<vec4>>([](Device &device) {
        auto out = BufferResource<vec4>{3};
        auto kernel = tests::UniformTypes{};
        kernel.a = -7;
        kernel.b = 9u;
        kernel.c = vec2{0.5f, -0.25f};
        kernel.d = ivec3{1, -2, 3};
        kernel.e = 42.0f;
        kernel.f = vec4{1.0f, 2.0f, 3.0f, 4.0f};
        kernel.out.attach(out);
        device.uploadBuffer(out);
        device.useKernel(kernel);
        device.bindBuffer(kernel.out);
        device.setUniform(kernel.a);
        device.setUniform(kernel.b);
        device.setUniform(kernel.c);
        device.setUniform(kernel.d);
        device.setUniform(kernel.e);
        device.setUniform(kernel.f);
        device.execute(kernel, uvec3{1u, 1u, 1u});
        device.downloadBuffer(out);
        return contents(out);
    });
    EXPECT_EQ(expected, actual);
}

TEST_P(Equivalence, Builtins) {
    auto const [expected, actual] = runBoth<std::vector<uvec4>>([](Device &device) {
        auto out = BufferResource<uvec4>{2 * 12 * 4};
        auto kernel = tests::Builtins{};
        kernel.out.attach(out);
        device.uploadBuffer(out);
        device.useKernel(kernel);
        device.bindBuffer(kernel.out);
        device.execute(kernel, uvec3{3u, 2u, 1u});
        device.downloadBuffer(out);
        return contents(out);
    });
    EXPECT_EQ(expected, actual);
}

TEST_P(Equivalence, ImageFormats) {
    struct Result {
        std::vector<cpu::RGBA8> colors;
        std::vector<cpu::RGB8> floats;
        std::vector<cpu::R32I> ints;
    };
    auto run = [](Device &device) {
        auto const size = ivec2{19, 13};
        auto colors = BufferResource<cpu::RGBA8, Dim::D2>{size};
        auto floats = BufferResource<cpu::RGB8, Dim::D2>{size};
        auto ints = BufferResource<cpu::R32I, Dim::D2>{size};
        for (int y = 0; y < size.y; ++y) {
            for (int x = 0; x < size.x; ++x) {
                auto const p = ivec2{x, y};
                colors[p] = cpu::RGBA8{{static_cast<std::uint8_t>(x * 13), static_cast<std::uint8_t>(y * 19),
                                        static_cast<std::uint8_t>(x * y), 200}};
                ints[p] = cpu::R32I{{x * 1000 - y * 77}};
            }
        }
        auto kernel = tests::ImageFormats{};
        kernel.colors.attach(colors);
        kernel.floats.attach(floats);
        kernel.signedInts.attach(ints);
        device.uploadImage(kernel.colors);
        device.uploadImage(kernel.floats);
        device.uploadImage(kernel.signedInts);
        device.useKernel(kernel);
        device.bindImage(kernel.colors);
        device.bindImage(kernel.floats);
        device.bindImage(kernel.signedInts);
        device.execute(kernel, uvec3{3u, 2u, 1u});
        device.downloadImage(kernel.colors);
        device.downloadImage(kernel.floats);
        device.downloadImage(kernel.signedInts);
        return Result{std::vector<cpu::RGBA8>(colors.begin(), colors.end()),
                      std::vector<cpu::RGB8>(floats.begin(), floats.end()),
                      std::vector<cpu::R32I>(ints.begin(), ints.end())};
    };
    try {
        auto const [expected, actual] = runBoth<Result>(run);
        for (std::size_t i = 0; i < expected.colors.size(); ++i) {
            for (int c = 0; c < 4; ++c) {
                EXPECT_NEAR(expected.colors[i].channels[c], actual.colors[i].channels[c], 1) << "texel " << i;
            }
            for (int c = 0; c < 3; ++c) {
                EXPECT_NEAR(expected.floats[i].channels[c], actual.floats[i].channels[c], 1) << "texel " << i;
            }
            EXPECT_EQ(expected.ints[i].channels[0], actual.ints[i].channels[0]) << "texel " << i;
        }
    } catch (SkipKernel const &skip) {
        GTEST_SKIP() << skip.reason;
    }
}

TEST_P(Equivalence, AtomicOps) {
    struct Result {
        std::vector<uint> bins;
        std::vector<uint> stats;
        std::vector<uint> compacted;
        std::vector<std::int32_t> slots;
    };
    auto const [expected, actual] = runBoth<Result>([](Device &device) {
        constexpr int n = 1000;
        auto values = BufferResource<uint>{n};
        auto seed = 99u;
        for (int i = 0; i < n; ++i) {
            seed = seed * 1664525u + 1013904223u;
            values[i] = seed >> 8;
        }
        auto bins = BufferResource<uint>{16};
        auto stats = BufferResource<uint>{7};
        stats[1] = 0xffffffffu;
        stats[3] = 0xffffffffu;
        auto compacted = BufferResource<uint>{n};
        auto slots = BufferResource<std::int32_t>{2 * n};
        for (int i = 0; i < n; ++i) {
            slots[2 * i + 1] = i;
        }
        auto kernel = tests::AtomicOps{};
        kernel.values.attach(values);
        kernel.bins.attach(bins);
        kernel.stats.attach(stats);
        kernel.compacted.attach(compacted);
        kernel.slots.attach(slots);
        for (auto *resource : {&values, &bins, &stats, &compacted}) {
            device.uploadBuffer(*resource);
        }
        device.uploadBuffer(slots);
        device.useKernel(kernel);
        device.bindBuffer(kernel.values);
        device.bindBuffer(kernel.bins);
        device.bindBuffer(kernel.stats);
        device.bindBuffer(kernel.compacted);
        device.bindBuffer(kernel.slots);
        device.execute(kernel, uvec3{(n + 63) / 64, 1u, 1u});
        device.downloadBuffer(bins);
        device.downloadBuffer(stats);
        device.downloadBuffer(compacted);
        device.downloadBuffer(slots);
        auto kept = std::vector<uint>(compacted.begin(), compacted.begin() + stats[5]);
        std::sort(kept.begin(), kept.end());
        return Result{contents(bins), contents(stats), kept, contents(slots)};
    });
    EXPECT_EQ(expected.bins, actual.bins);
    EXPECT_EQ(expected.stats, actual.stats);
    EXPECT_EQ(expected.compacted, actual.compacted);
    EXPECT_EQ(expected.slots, actual.slots);
}

std::vector<uint> randomValues(int const n, uint seed) {
    auto values = std::vector<uint>(static_cast<std::size_t>(n));
    for (auto &value : values) {
        seed = seed * 1664525u + 1013904223u;
        value = seed >> 12;
    }
    return values;
}

TEST_P(Equivalence, WorkgroupReduce) {
    auto const [expected, actual] = runBoth<std::vector<uint>>([](Device &device) {
        constexpr int n = 1000;
        auto values = BufferResource<uint>{n};
        auto const source = randomValues(n, 5u);
        std::copy(source.begin(), source.end(), values.begin());
        auto sums = BufferResource<uint>{(n + 63) / 64};
        auto kernel = tests::WorkgroupReduce{};
        kernel.values.attach(values);
        kernel.sums.attach(sums);
        device.uploadBuffer(values);
        device.uploadBuffer(sums);
        device.useKernel(kernel);
        device.bindBuffer(kernel.values);
        device.bindBuffer(kernel.sums);
        device.execute(kernel, uvec3{static_cast<uint>(sums.size()), 1u, 1u});
        device.downloadBuffer(sums);
        return contents(sums);
    });
    EXPECT_EQ(expected, actual);
    auto const source = randomValues(1000, 5u);
    auto total = uint{0};
    for (auto const value : source) {
        total += value;
    }
    auto reduced = uint{0};
    for (auto const sum : actual) {
        reduced += sum;
    }
    EXPECT_EQ(reduced, total);
}

TEST_P(Equivalence, TiledTranspose) {
    auto const [expected, actual] = runBoth<std::vector<float>>([](Device &device) {
        auto const size = ivec2{37, 23};
        auto input = BufferResource<float>{size.x * size.y};
        for (int i = 0; i < size.x * size.y; ++i) {
            input[i] = static_cast<float>(i);
        }
        auto output = BufferResource<float>{size.x * size.y};
        auto kernel = tests::TiledTranspose{};
        kernel.size = size;
        kernel.input.attach(input);
        kernel.output.attach(output);
        device.uploadBuffer(input);
        device.uploadBuffer(output);
        device.useKernel(kernel);
        device.bindBuffer(kernel.input);
        device.bindBuffer(kernel.output);
        device.setUniform(kernel.size);
        device.execute(kernel, uvec3{(37u + 15u) / 16u, (23u + 15u) / 16u, 1u});
        device.downloadBuffer(output);
        return contents(output);
    });
    EXPECT_EQ(expected, actual);
    EXPECT_FLOAT_EQ(actual[5 * 23 + 7], static_cast<float>(7 * 37 + 5)); // output(y=5, x=7) = input(y=7, x=5)
}

TEST_P(Equivalence, SharedHistogram) {
    auto const [expected, actual] = runBoth<std::vector<uint>>([](Device &device) {
        constexpr int n = 3000;
        auto values = BufferResource<uint>{n};
        auto const source = randomValues(n, 11u);
        std::copy(source.begin(), source.end(), values.begin());
        auto bins = BufferResource<uint>{33};
        auto kernel = tests::SharedHistogram{};
        kernel.values.attach(values);
        kernel.bins.attach(bins);
        device.uploadBuffer(values);
        device.uploadBuffer(bins);
        device.useKernel(kernel);
        device.bindBuffer(kernel.values);
        device.bindBuffer(kernel.bins);
        device.execute(kernel, uvec3{(n + 127) / 128, 1u, 1u});
        device.downloadBuffer(bins);
        return contents(bins);
    });
    EXPECT_EQ(expected, actual);
    EXPECT_EQ(actual[32], 3000u);
}

TEST_P(Equivalence, MatrixOps) {
    struct Result {
        std::vector<mat3> mats;
        std::vector<vec4> out;
    };
    auto const [expected, actual] = runBoth<Result>([](Device &device) {
        constexpr int n = 40;
        auto mats = BufferResource<mat3>{n};
        for (int i = 0; i < n; ++i) {
            auto const f = static_cast<float>(i);
            mats[i] = mat3{2.0f + f, 1.0f, 0.5f, 0.0f, 3.0f, f * 0.1f, 1.0f, 0.25f, 4.0f};
        }
        auto out = BufferResource<vec4>{2 * n};
        auto kernel = tests::MatrixOps{};
        kernel.colorMatrix = mat3{1.0f, 0.5f, 0.0f, -1.0f, 2.0f, 0.25f, 0.0f, 0.75f, 1.5f};
        kernel.projection = mat2x3{vec3{1.0f, 2.0f, 3.0f}, vec3{-1.0f, 0.5f, 4.0f}};
        kernel.mats.attach(mats);
        kernel.out.attach(out);
        device.uploadBuffer(mats);
        device.uploadBuffer(out);
        device.useKernel(kernel);
        device.bindBuffer(kernel.mats);
        device.bindBuffer(kernel.out);
        device.setUniform(kernel.colorMatrix);
        device.setUniform(kernel.projection);
        device.execute(kernel, uvec3{3u, 1u, 1u});
        device.downloadBuffer(mats);
        device.downloadBuffer(out);
        return Result{contents(mats), contents(out)};
    });
    for (std::size_t i = 0; i < expected.mats.size(); ++i) {
        for (int c = 0; c < 3; ++c) {
            for (int r = 0; r < 3; ++r) {
                EXPECT_NEAR(expected.mats[i][c][r], actual.mats[i][c][r], 1e-4f) << "matrix " << i;
            }
        }
    }
    for (std::size_t i = 0; i < expected.out.size(); ++i) {
        for (int c = 0; c < 4; ++c) {
            EXPECT_NEAR(expected.out[i][c], actual.out[i][c], 1e-3f) << "element " << i << " component " << c;
        }
    }
}

TEST_P(Equivalence, PackingOps) {
    auto const [expected, actual] = runBoth<std::vector<uvec4>>([](Device &device) {
        constexpr int n = 64;
        auto words = BufferResource<uint>{n};
        auto seed = 12345u;
        for (int i = 0; i < n; ++i) {
            seed = seed * 1664525u + 1013904223u;
            words[i] = seed;
        }
        auto out = BufferResource<uvec4>{2 * n};
        auto kernel = tests::PackingOps{};
        kernel.words.attach(words);
        kernel.out.attach(out);
        device.uploadBuffer(words);
        device.uploadBuffer(out);
        device.useKernel(kernel);
        device.bindBuffer(kernel.words);
        device.bindBuffer(kernel.out);
        device.execute(kernel, uvec3{n / 32, 1u, 1u});
        device.downloadBuffer(out);
        return contents(out);
    });
    EXPECT_EQ(expected, actual);
}

TEST_P(Equivalence, TextureSampling) {
    auto const [expected, actual] = runBoth<std::vector<vec4>>([](Device &device) {
        auto const size = ivec2{13, 9};
        auto gray = BufferResource<cpu::R8, Dim::D2>{size};
        auto color = BufferResource<cpu::RGBA8, Dim::D2>{size};
        auto ids = BufferResource<cpu::R32UI, Dim::D2>{size};
        for (int y = 0; y < size.y; ++y) {
            for (int x = 0; x < size.x; ++x) {
                auto const p = ivec2{x, y};
                gray[p] = cpu::R8{{static_cast<std::uint8_t>(x * 19 + y * 7)}};
                color[p] = cpu::RGBA8{{static_cast<std::uint8_t>(x * 20), static_cast<std::uint8_t>(y * 28),
                                       static_cast<std::uint8_t>((x + y) * 9), 255}};
                ids[p] = cpu::R32UI{{static_cast<uint>(x * 100 + y)}};
            }
        }
        auto out = BufferResource<vec4>{3 * size.x * size.y};
        auto kernel = tests::TextureSampling{};
        kernel.gray.attach(gray);
        kernel.color.attach(color);
        kernel.ids.attach(ids);
        kernel.out.attach(out);
        device.uploadImage(kernel.gray);
        device.uploadImage(kernel.color);
        device.uploadImage(kernel.ids);
        device.uploadBuffer(out);
        device.useKernel(kernel);
        device.bindTexture(kernel.gray);
        device.bindTexture(kernel.color);
        device.bindTexture(kernel.ids);
        device.bindBuffer(kernel.out);
        device.execute(kernel, uvec3{2u, 2u, 1u});
        device.downloadBuffer(out);
        return contents(out);
    });
    for (std::size_t i = 0; i < expected.size(); ++i) {
        // Bilinear weights are quantised on GPUs (typically 8 fractional bits).
        auto const tolerance = i % 3 == 1 ? 2.0f / 255.0f : 1e-5f;
        for (int c = 0; c < 4; ++c) {
            EXPECT_NEAR(expected[i][c], actual[i][c], tolerance) << "element " << i << " component " << c;
        }
    }
}

TEST_P(Equivalence, Nv12ToRgba) {
    auto const [expected, actual] = runBoth<std::vector<cpu::RGBA8>>([](Device &device) {
        auto frame = examples::makeTestFrame(ivec2{96, 54});
        auto const rgba = examples::convertNv12(device, frame, examples::bt601Limited());
        return std::vector<cpu::RGBA8>(rgba.begin(), rgba.end());
    });
    for (std::size_t i = 0; i < expected.size(); ++i) {
        for (int c = 0; c < 4; ++c) {
            EXPECT_NEAR(expected[i].channels[c], actual[i].channels[c], 1) << "pixel " << i;
        }
    }
}

TEST_P(Equivalence, GameOfLife) {
    auto const initial = examples::randomGrid(61, 7u);
    auto const [expected, actual] = runBoth<std::vector<cpu::R8UI>>([&initial](Device &device) {
        auto const grid = examples::simulate(device, initial, 25);
        return std::vector<cpu::R8UI>(grid.begin(), grid.end());
    });
    auto mismatches = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        mismatches += expected[i].channels[0] != actual[i].channels[0] ? 1 : 0;
    }
    EXPECT_EQ(mismatches, 0);
}

TEST_P(Equivalence, Raytracer) {
    auto const [expected, actual] = runBoth<std::vector<cpu::RGBA8>>([](Device &device) {
        auto const picture = examples::render(device, ivec2{160, 120}, 0.6f);
        return std::vector<cpu::RGBA8>(picture.begin(), picture.end());
    });
    auto total = 0.0;
    auto farOff = 0;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        for (int c = 0; c < 3; ++c) {
            auto const diff = std::abs(int{expected[i].channels[c]} - int{actual[i].channels[c]});
            total += diff;
            farOff += diff > 16 ? 1 : 0;
        }
    }
    // GPU transcendental functions are less precise than the host's; a few
    // pixels on silhouettes may flip, but the image must match.
    EXPECT_LT(total / (expected.size() * 3.0), 0.5);
    EXPECT_LT(farOff, static_cast<int>(expected.size() / 500));
}

INSTANTIATE_TEST_SUITE_P(Backends, Equivalence, testing::ValuesIn(gpuCases()), caseName);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(Equivalence);

} // namespace
