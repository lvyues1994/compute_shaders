// Runs the same kernels on the CPU backend and on every GPU backend that can
// be created on this machine, and checks that the results agree.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <game_of_life/simulate.hpp>
#include <nv12_to_rgba/convert.hpp>
#include <raytracer/render.hpp>

#include "equivalence.hpp"
#include "kernels/feature_kernels.hpp"

namespace {

using namespace tc;
using namespace tc::testing_support;

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

struct TableState {
    std::vector<tests::Spark> sparks;
    std::vector<tests::Probe> probes;
    std::vector<vec3> points;
    std::vector<vec4> report;
};

tests::Spark makeSpark(int const i) {
    auto const f = static_cast<float>(i);
    return tests::Spark{vec3{f, 2.0f * f, -f}, 1.0f + f, vec3{0.5f, f, 1.0f}, static_cast<uint>(i % 3)};
}

tests::Probe makeProbe(int const i) {
    auto const f = static_cast<float>(i);
    auto probe = tests::Probe{};
    probe.frame = mat3{vec3{1.0f, f, 0.0f}, vec3{0.0f, 1.0f, f}, vec3{f, 0.0f, 1.0f}};
    probe.weights = {f, 0.5f, -f};
    probe.spark = makeSpark(i + 100);
    probe.active = i % 2 == 0;
    probe.phase = tests::Phase::Idle;
    probe.counter = 1000 + i;
    return probe;
}

template <class Resource>
struct TableTag {
    using type = Resource;
};

// Runs TableOps or TableOpsSoa on tables from `makeTable` (host or device
// memory).
template <class Kernel, class MakeTable>
TableState runTableOps(Device &device, MakeTable const &makeTable) {
    constexpr int n = 50;
    auto sparks = makeTable(TableTag<typename decltype(Kernel::sparks)::Resource>{}, n);
    auto probes = makeTable(TableTag<typename decltype(Kernel::probes)::Resource>{}, n);
    auto points = makeTable(TableTag<typename decltype(Kernel::points)::Resource>{}, n);
    for (int i = 0; i < n; ++i) {
        sparks.set(i, makeSpark(i));
        probes.set(i, makeProbe(i));
        points.set(i, vec3{static_cast<float>(i), 1.0f, -2.0f});
    }
    auto report = BufferResource<vec4>{n};
    auto kernel = Kernel{};
    kernel.sparks.attach(sparks);
    kernel.probes.attach(probes);
    kernel.points.attach(points);
    kernel.report.attach(report);
    kernel.dt = 0.5f;
    device.uploadTable(sparks);
    device.uploadTable(probes);
    device.uploadTable(points);
    device.uploadBuffer(report);
    device.useKernel(kernel);
    device.bindTable(kernel.sparks);
    device.bindTable(kernel.probes);
    device.bindTable(kernel.points);
    device.bindBuffer(kernel.report);
    device.setUniform(kernel.dt);
    device.execute(kernel, uvec3{2u, 1u, 1u});
    device.downloadTable(sparks);
    device.downloadTable(probes);
    device.downloadTable(points);
    device.downloadBuffer(report);
    auto state = TableState{};
    for (int i = 0; i < n; ++i) {
        state.sparks.push_back(sparks.get(i));
        state.probes.push_back(probes.get(i));
        state.points.push_back(points.get(i));
    }
    state.report = contents(report);
    return state;
}

void expectNear(vec3 const a, vec3 const b, std::size_t const i) {
    for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(a[c], b[c], 1e-4f) << "element " << i << " component " << c;
    }
}

void expectSameTables(TableState const &expected, TableState const &actual) {
    ASSERT_EQ(expected.sparks.size(), actual.sparks.size());
    for (std::size_t i = 0; i < expected.sparks.size(); ++i) {
        auto const &e = expected.sparks[i];
        auto const &a = actual.sparks[i];
        expectNear(e.position, a.position, i);
        expectNear(e.velocity, a.velocity, i);
        EXPECT_FLOAT_EQ(e.mass, a.mass) << i;
        EXPECT_EQ(e.flags, a.flags) << i;
        auto const &ep = expected.probes[i];
        auto const &ap = actual.probes[i];
        for (int c = 0; c < 3; ++c) {
            expectNear(ep.frame[c], ap.frame[c], i);
        }
        EXPECT_EQ(ep.weights, ap.weights) << i;
        expectNear(ep.spark.velocity, ap.spark.velocity, i);
        EXPECT_EQ(ep.spark.flags, ap.spark.flags) << i;
        EXPECT_EQ(ep.active, ap.active) << i;
        EXPECT_EQ(ep.phase, ap.phase) << i;
        EXPECT_EQ(ep.counter, ap.counter) << i;
        expectNear(expected.points[i], actual.points[i], i);
        for (int c = 0; c < 4; ++c) {
            EXPECT_NEAR(expected.report[i][c], actual.report[i][c], 1e-4f) << i;
        }
    }
}

auto const hostTables = [](auto tag, int const n) {
    return typename decltype(tag)::type(static_cast<std::size_t>(n));
};

TEST_P(Equivalence, PackedTables) {
    auto const [expected, actual] =
        runBoth<TableState>([&](Device &device) { return runTableOps<tests::TableOps>(device, hostTables); });
    expectSameTables(expected, actual);
    // A spot check against hand-computed values: spark 1 takes the update
    // branch (flags 1), spark 3 the field-edit branch (flags 0).
    EXPECT_NEAR(actual.sparks[1].velocity.y, 1.0f - 9.81f * 0.5f, 1e-4f);
    EXPECT_EQ(actual.sparks[3].flags, 1u);
    EXPECT_FLOAT_EQ(actual.sparks[3].mass, 2.0f);
    EXPECT_EQ(actual.probes[2].phase, tests::Phase::Resting);
    EXPECT_FLOAT_EQ(actual.probes[2].weights[2], 7.0f);
    EXPECT_FLOAT_EQ(actual.report[0].w, 50.0f);
}

TEST_P(Equivalence, ZeroCopyTables) {
    auto const [expected, actual] = runBoth<TableState>([](Device &device) {
        auto const deviceTables = [&device](auto tag, int const n) {
            using Resource = typename decltype(tag)::type;
            return device.createTable<typename Resource::value_type, typename Resource::layout_type>(
                static_cast<std::size_t>(n));
        };
        return runTableOps<tests::TableOps>(device, deviceTables);
    });
    expectSameTables(expected, actual);
    if (GetParam().backend == "vulkan") {
        EXPECT_TRUE(gpu->createTable<tests::Spark>(4).usesExternalMemory());
    }
}

// Switching the layout changes neither the kernel code nor the results.
TEST_P(Equivalence, SoaTables) {
    auto const [expected, actual] =
        runBoth<TableState>([&](Device &device) { return runTableOps<tests::TableOpsSoa>(device, hostTables); });
    expectSameTables(expected, actual);
    auto const aos = runTableOps<tests::TableOps>(*cpu, hostTables);
    expectSameTables(aos, actual);
}

// A layout defined in user code: its fieldAddress runs on the host and,
// translated, on the GPU.
TEST_P(Equivalence, UserLayoutTables) {
    auto const [expected, actual] =
        runBoth<TableState>([&](Device &device) { return runTableOps<tests::TableOpsPadded>(device, hostTables); });
    expectSameTables(expected, actual);
    auto const aos = runTableOps<tests::TableOps>(*cpu, hostTables);
    expectSameTables(aos, actual);
    EXPECT_EQ((TableResource<tests::Probe, tests::PaddedAos>::wordsFor(50)), 4u + 50u * 24u);
}

struct CodecState {
    std::vector<tests::Surfel> surfels;
    std::vector<tests::Surfel> copies;
    std::vector<vec4> report;
};

tests::Surfel makeSurfel(int const i) {
    auto const f = static_cast<float>(i);
    auto surfel = tests::Surfel{};
    surfel.transform = mat4{vec4{1.0f, 0.0f, 0.0f, 0.0f}, vec4{0.0f, 0.5f, 0.0f, 0.0f}, vec4{0.0f, 0.0f, 2.0f, 0.0f},
                            vec4{f, -f, 0.25f * f, 1.0f}};
    surfel.color = vec4{0.1f, 0.5f, 0.9f, f / 64.0f};
    surfel.normal = normalize(vec3{f - 20.0f, 3.0f, -2.0f});
    surfel.uv = vec2{f / 7.0f, 0.3f};
    surfel.extra = vec4{f * 1.5f, -2.0f, 1000.0f, 0.001f};
    surfel.albedo = vec3{0.2f, f / 64.0f, 0.7f};
    surfel.tangent = vec4{-0.3f, 0.6f, 0.0f, 1.0f};
    surfel.weight = 1.0f + f;
    return surfel;
}

CodecState runCodecOps(Device &device) {
    constexpr int n = 40;
    auto surfels = TableResource<tests::Surfel>{n};
    auto copies = TableResource<tests::Surfel, Soa<16>>{n};
    for (int i = 0; i < n; ++i) {
        surfels.set(i, makeSurfel(i));
    }
    auto report = BufferResource<vec4>{4 * n};
    auto kernel = tests::CodecOps{};
    kernel.surfels.attach(surfels);
    kernel.copies.attach(copies);
    kernel.report.attach(report);
    device.uploadTable(surfels);
    device.uploadTable(copies);
    device.uploadBuffer(report);
    device.useKernel(kernel);
    device.bindTable(kernel.surfels);
    device.bindTable(kernel.copies);
    device.bindBuffer(kernel.report);
    device.execute(kernel, uvec3{2u, 1u, 1u});
    device.downloadTable(surfels);
    device.downloadTable(copies);
    device.downloadBuffer(report);
    auto state = CodecState{};
    for (int i = 0; i < n; ++i) {
        state.surfels.push_back(surfels.get(i));
        state.copies.push_back(copies.get(i));
    }
    state.report = contents(report);
    return state;
}

template <class V>
void expectNearVec(V const &a, V const &b, float const tolerance, std::string const &what) {
    for (int c = 0; c < V::length(); ++c) {
        EXPECT_NEAR(a[c], b[c], tolerance * std::max(1.0f, std::abs(a[c]))) << what << " component " << c;
    }
}

// Half floats and 8-bit channels may round differently on GPUs: one step.
void expectSameSurfel(tests::Surfel const &e, tests::Surfel const &a, std::string const &what) {
    for (int c = 0; c < 4; ++c) {
        expectNearVec(e.transform.value[c], a.transform.value[c], 1e-6f, what + " transform");
    }
    expectNearVec(e.color.value, a.color.value, 1.01f / 255.0f, what + " color");
    expectNearVec(e.normal.value, a.normal.value, 2e-4f, what + " normal");
    expectNearVec(e.uv.value, a.uv.value, 1e-3f, what + " uv");
    expectNearVec(e.extra.value, a.extra.value, 1e-3f, what + " extra");
    expectNearVec(e.albedo.value, a.albedo.value, 1.01f / 1023.0f, what + " albedo");
    expectNearVec(e.tangent.value, a.tangent.value, 1.01f / 127.0f, what + " tangent");
    EXPECT_FLOAT_EQ(e.weight, a.weight) << what;
}

TEST_P(Equivalence, Codecs) {
    auto const [expected, actual] = runBoth<CodecState>(runCodecOps);
    for (std::size_t i = 0; i < expected.surfels.size(); ++i) {
        expectSameSurfel(expected.surfels[i], actual.surfels[i], "surfel " + std::to_string(i));
        expectSameSurfel(expected.copies[i], actual.copies[i], "copy " + std::to_string(i));
        for (int r = 0; r < 4; ++r) {
            expectNearVec(expected.report[4 * i + r], actual.report[4 * i + r], 2e-3f,
                          "report " + std::to_string(4 * i + r));
        }
    }
    // What the codecs keep: 8-bit colour, unit normals, exact affine parts.
    auto const original = makeSurfel(5);
    EXPECT_NEAR(actual.copies[5].color.value.x, 1.0f - original.color.value.x, 1.0f / 255.0f);
    EXPECT_NEAR(length(actual.surfels[5].normal.value), 1.0f, 1e-5f);
    EXPECT_EQ(actual.surfels[5].transform.value[3], original.transform.value[3]);
    EXPECT_NEAR(actual.surfels[5].uv.value.y, 0.3f + 0.25f, 1e-3f);
    EXPECT_NE(actual.surfels[5].uv.value.y, 0.3f + 0.25f) << "stored as half precision";
}

TEST_P(Equivalence, LanguageFeatures) {
    auto const [expected, actual] = runBoth<std::vector<vec4>>([](Device &device) {
        auto out = BufferResource<vec4>{64};
        auto kernel = tests::LanguageFeatures{};
        kernel.out.attach(out);
        device.uploadBuffer(out);
        device.useKernel(kernel);
        device.bindBuffer(kernel.out);
        device.execute(kernel, uvec3{1u, 1u, 1u});
        device.downloadBuffer(out);
        return contents(out);
    });
    for (std::size_t i = 0; i < expected.size(); ++i) {
        expectNearVec(expected[i], actual[i], 1e-5f, "element " + std::to_string(i));
    }
    // i = 3: z = 0.3 + i, z * z = -0.91 + 0.6i, w = (z * z - z) / 2 = -0.605 - 0.2i.
    EXPECT_NEAR(actual[6].x, -0.605f, 1e-5f);
    EXPECT_NEAR(actual[6].z, -0.2f, 1e-5f);
    EXPECT_EQ(actual[7], (vec4{6.0f, 9.0f, 6.0f, 6.0f}));
}

// One timing per execute(), in order, while timers are enabled.
TEST_P(Equivalence, Timers) {
    if (not gpu->info().timers) {
        GTEST_SKIP() << "no timer queries on " << gpu->info().deviceName;
    }
    for (auto *device : {cpu.get(), gpu.get()}) {
        constexpr int n = 1 << 16;
        auto values = BufferResource<uint>{n};
        auto sums = BufferResource<uint>{n / 64};
        auto kernel = tests::WorkgroupReduce{};
        kernel.values.attach(values);
        kernel.sums.attach(sums);
        device->uploadBuffer(values);
        device->uploadBuffer(sums);
        device->useKernel(kernel);
        device->bindBuffer(kernel.values);
        device->bindBuffer(kernel.sums);
        device->execute(kernel, uvec3{n / 64u, 1u, 1u}); // not timed
        device->enableTimers(true);
        for (uint groups = 1; groups <= 3; ++groups) {
            device->execute(kernel, uvec3{groups * 256u, 1u, 1u});
        }
        auto const timings = device->takeTimings();
        ASSERT_EQ(timings.size(), 3u) << device->info().backend;
        auto total = 0.0;
        for (uint i = 0; i < 3; ++i) {
            EXPECT_EQ(timings[i].program, "workgroup_reduce");
            EXPECT_EQ(timings[i].workGroups, (uvec3{(i + 1) * 256u, 1u, 1u}));
            if (timings[i].milliseconds) {
                EXPECT_GE(*timings[i].milliseconds, 0.0);
                EXPECT_LT(*timings[i].milliseconds, 10'000.0);
                total += *timings[i].milliseconds;
            }
        }
        EXPECT_GT(total, 0.0) << device->info().backend;
        device->enableTimers(false);
        device->execute(kernel, uvec3{1u, 1u, 1u});
        EXPECT_TRUE(device->takeTimings().empty());
    }
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

} // namespace

namespace tc::testing_support {

// Instantiates the Equivalence cases of every test file.
INSTANTIATE_TEST_SUITE_P(Backends, Equivalence, testing::ValuesIn(gpuCases()), caseName);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(Equivalence);

} // namespace tc::testing_support
