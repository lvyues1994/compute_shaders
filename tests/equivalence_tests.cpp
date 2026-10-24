// Runs the same kernels on the CPU backend and on every GPU backend that can
// be created on this machine, and checks that the results agree.

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
