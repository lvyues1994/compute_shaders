// Kernel templates, indirect dispatch and buffer ranges on every backend.

#include <algorithm>
#include <cmath>
#include <vector>

#include <tc/program.hpp>

#include "equivalence.hpp"
#include "kernels/dispatch_kernels.hpp"

namespace {

using namespace tc;
using namespace tc::testing_support;

template <class Kernel>
std::vector<vec2> runShade(Device &device, float const exposure) {
    constexpr int n = 200;
    auto nDotL = BufferResource<float>{n};
    for (int i = 0; i < n; ++i) {
        nDotL[i] = std::sin(static_cast<float>(i) * 0.1f);
    }
    auto out = BufferResource<vec2>{n};
    auto kernel = Kernel{};
    kernel.nDotL.attach(nDotL);
    kernel.out.attach(out);
    kernel.exposure = exposure;
    device.uploadBuffer(nDotL);
    device.uploadBuffer(out);
    device.useKernel(kernel);
    device.bindBuffer(kernel.nDotL);
    device.bindBuffer(kernel.out);
    device.setUniform(kernel.exposure);
    auto const groupSize = uvec3{kernel.local_size}.x;
    device.execute(kernel, uvec3{(n + groupSize - 1) / groupSize, 1u, 1u});
    device.downloadBuffer(out);
    return contents(out);
}

// One kernel template, two GLSL programs; each instance behaves like the
// C++ it was instantiated from.
TEST_P(Equivalence, KernelTemplates) {
    auto const [lambertCpu, lambertGpu] = runBoth<std::vector<vec2>>(
        [](Device &device) { return runShade<tests::Shade<tests::Lambert>>(device, 1.5f); });
    auto const [toonCpu, toonGpu] =
        runBoth<std::vector<vec2>>([](Device &device) { return runShade<tests::Shade<tests::Toon, 64>>(device, 2.0f); });
    for (std::size_t i = 0; i < lambertCpu.size(); ++i) {
        EXPECT_NEAR(lambertCpu[i].x, lambertGpu[i].x, 1e-5f) << i;
        EXPECT_NEAR(toonCpu[i].x, toonGpu[i].x, 1e-5f) << i;
        EXPECT_EQ(lambertGpu[i].y, 32.0f);
        EXPECT_EQ(toonGpu[i].y, 64.0f);
    }
    // sin(0.5) = 0.479: Lambert keeps it, Toon rounds down to a third.
    EXPECT_NEAR(lambertGpu[5].x, 0.4794f * 1.5f + 0.2f, 1e-3f);
    EXPECT_NEAR(toonGpu[5].x, (1.0f / 3.0f + 0.01f) * 2.0f + 0.4f, 1e-4f);
}

struct CullResult {
    std::vector<float> shaded;
    std::vector<uint> visible; // sorted: atomics hand out slots in any order
    DispatchIndirectCommand args;
};

CullResult cullAndShade(Device &device) {
    constexpr int n = 1000;
    auto spheres = BufferResource<vec4>{n};
    auto seed = 7u;
    for (int i = 0; i < n; ++i) {
        seed = seed * 1664525u + 1013904223u;
        auto const x = static_cast<float>(seed >> 8) / 16777216.0f * 20.0f - 10.0f;
        spheres[i] = vec4{x, static_cast<float>(i % 7), 0.0f, 0.25f + static_cast<float>(i % 3)};
    }
    auto visible = BufferResource<uint>{n};
    auto count = BufferResource<uint>{1};
    auto args = BufferResource<DispatchIndirectCommand>{1};
    auto out = BufferResource<float>{n};
    for (auto *buffer : {&visible, &count}) {
        device.uploadBuffer(*buffer);
    }
    device.uploadBuffer(spheres);
    device.uploadBuffer(args);
    device.uploadBuffer(out);

    auto cull = tests::CullSpheres{};
    cull.spheres.attach(spheres);
    cull.visible.attach(visible);
    cull.count.attach(count);
    cull.args.attach(args);
    cull.plane = vec4{1.0f, 0.0f, 0.0f, -2.0f};
    device.useKernel(cull);
    device.bindBuffer(cull.spheres);
    device.bindBuffer(cull.visible);
    device.bindBuffer(cull.count);
    device.bindBuffer(cull.args);
    device.setUniform(cull.plane);
    device.execute(cull, uvec3{(n + 63) / 64, 1u, 1u});

    auto shade = tests::ShadeVisible{};
    shade.spheres.attach(spheres);
    shade.visible.attach(visible);
    shade.count.attach(count);
    shade.out.attach(out);
    device.useKernel(shade);
    device.bindBuffer(shade.spheres);
    device.bindBuffer(shade.visible);
    device.bindBuffer(shade.count);
    device.bindBuffer(shade.out);
    device.executeIndirect(shade, args);

    device.downloadBuffer(out);
    device.downloadBuffer(visible);
    device.downloadBuffer(count);
    device.downloadBuffer(args);
    auto kept = std::vector<uint>(visible.begin(), visible.begin() + count[0]);
    std::sort(kept.begin(), kept.end());
    return CullResult{contents(out), kept, args[0]};
}

// A kernel decides how many workgroups the next one runs.
TEST_P(Equivalence, IndirectDispatchFromKernel) {
    auto const [expected, actual] = runBoth<CullResult>(cullAndShade);
    EXPECT_EQ(expected.visible, actual.visible);
    EXPECT_EQ(expected.shaded, actual.shaded);
    auto const visibleCount = static_cast<uint>(actual.visible.size());
    EXPECT_GT(visibleCount, 100u);
    EXPECT_LT(visibleCount, 900u);
    EXPECT_EQ(actual.args.x, (visibleCount + 63u) / 64u);
    EXPECT_EQ(actual.args.y, 1u);
    EXPECT_EQ(actual.shaded[actual.visible.front()],
              2.0f * (0.25f + static_cast<float>(actual.visible.front() % 3)) + static_cast<float>(actual.args.x));
}

// Host-written counts in three dimensions, read from the second command.
TEST_P(Equivalence, IndirectDispatchFromHost) {
    auto timed = std::vector<KernelTiming>{};
    auto const [expected, actual] = runBoth<std::vector<uvec4>>([&timed](Device &device) {
        timed.clear(); // keeps the GPU's timings, which run second
        auto args = BufferResource<DispatchIndirectCommand>{2};
        args[0] = DispatchIndirectCommand{1000u, 1000u, 1000u};
        args[1] = DispatchIndirectCommand{3u, 2u, 2u};
        auto out = BufferResource<uvec4>{3 * 4 * 2 * 2 * 2};
        auto kernel = tests::GridIds{};
        kernel.out.attach(out);
        device.uploadBuffer(args);
        device.uploadBuffer(out);
        device.useKernel(kernel);
        device.bindBuffer(kernel.out);
        device.enableTimers(device.info().timers);
        device.executeIndirect(kernel, args, 1);
        device.downloadBuffer(out);
        if (device.info().timers) {
            timed = device.takeTimings();
            device.enableTimers(false);
        }
        return contents(out);
    });
    EXPECT_EQ(expected, actual);
    EXPECT_EQ(actual[5], (uvec4{5u, 0u, 0u, 322u}));
    EXPECT_EQ(actual.back(), (uvec4{11u, 3u, 1u, 322u}));
    if (not timed.empty()) {
        EXPECT_TRUE(timed.front().indirect);
        EXPECT_EQ(timed.front().workGroups, uvec3{}) << "GPU backends do not know indirect counts";
    }
}

TEST(IndirectDispatch, CpuReportsTheCountsItRan) {
    auto device = cpu::createDevice();
    auto args = BufferResource<DispatchIndirectCommand>{1};
    args[0] = DispatchIndirectCommand{2u, 1u, 1u};
    auto out = BufferResource<uvec4>{16};
    auto kernel = tests::GridIds{};
    kernel.out.attach(out);
    device->enableTimers(true);
    device->executeIndirect(kernel, args);
    auto const timings = device->takeTimings();
    ASSERT_EQ(timings.size(), 1u);
    EXPECT_TRUE(timings[0].indirect);
    EXPECT_EQ(timings[0].workGroups, (uvec3{2u, 1u, 1u}));
    EXPECT_THROW(device->executeIndirect(kernel, args, 1), DeviceError);
}

// The minimums Vulkan and OpenGL ES 3.1 guarantee.
TEST_P(Equivalence, DeviceLimits) {
    for (auto *device : {cpu.get(), gpu.get()}) {
        auto const limits = device->info().limits;
        auto const name = device->info().backend;
        EXPECT_GE(limits.storageBuffers, 4u) << name;
        EXPECT_GE(limits.storageImages, 4u) << name;
        EXPECT_GE(limits.sampledTextures, 16u) << name;
        EXPECT_GE(limits.storageBufferBytes, std::uint64_t{1} << 27) << name;
        EXPECT_GE(limits.sharedBytes, 16384u) << name;
        EXPECT_GE(limits.workGroupInvocations, 128u) << name;
        EXPECT_GE(limits.workGroupSize.x, 128u) << name;
        EXPECT_GE(limits.workGroupCount.x, 65535u) << name;
        EXPECT_GE(limits.storageBufferOffsetAlignment, 1u) << name;
        EXPECT_LE(limits.storageBufferOffsetAlignment, 256u) << name;
        if (name == "vulkan") {
            ASSERT_TRUE(limits.pushConstantBytes.has_value());
            EXPECT_GE(*limits.pushConstantBytes, 128u);
        }
    }
}

// What an engine does at startup: every transpiled kernel either runs here
// or says why not.
TEST_P(Equivalence, CheckEveryKernel) {
    auto const programs = ProgramRegistry::instance().all();
    EXPECT_GE(programs.size(), 30u);
    for (auto const &program : programs) {
        EXPECT_EQ(cpu->checkKernel(program.name), std::nullopt) << program.name;
        auto const reason = gpu->checkKernel(program.name);
        if (program.name == "image_formats" and GetParam().backend == "gles") {
            ASSERT_TRUE(reason.has_value());
            EXPECT_NE(reason->find("cannot run on OpenGL ES"), std::string::npos) << *reason;
        } else if (reason) {
            EXPECT_NE(reason->find("does not support"), std::string::npos) << *reason;
        }
    }
    EXPECT_NE(gpu->checkKernel("no_such_kernel"), std::nullopt);
}

TEST(DeviceLimits, KernelsOverALimitAreReported) {
    auto resources = std::vector<ResourceSlot>(5);
    for (std::uint32_t i = 0; i < 5; ++i) {
        resources[i].binding = i;
    }
    resources[4].kind = ResourceKind::StorageImage;
    auto program = KernelProgram{};
    program.name = "big";
    program.localSize = uvec3{64u, 4u, 1u};
    program.resources = Span<ResourceSlot const>{resources.data(), resources.size()};
    program.sharedBytes = 20000;
    auto limits = DeviceLimits{};
    limits.storageBuffers = 8;
    limits.storageImages = 4;
    limits.sampledTextures = 16;
    limits.sharedBytes = 32768;
    limits.workGroupInvocations = 256;
    limits.workGroupSize = uvec3{256u, 256u, 64u};
    EXPECT_EQ(exceedsLimits(program, limits), std::nullopt);
    limits.storageBuffers = 3;
    EXPECT_EQ(exceedsLimits(program, limits).value_or(""),
              "kernel 'big' uses 4 storage buffers; this device allows 3");
    limits.storageBuffers = 4;
    limits.sharedBytes = 16384;
    EXPECT_NE(exceedsLimits(program, limits).value_or("").find("shared memory"), std::string::npos);
    limits.sharedBytes = 32768;
    program.features = static_cast<std::uint32_t>(KernelFeature::Double);
    EXPECT_NE(exceedsLimits(program, limits).value_or("").find("double precision"), std::string::npos);
}

constexpr int kFrameElements = 64; // 1 KiB of vec4: aligned for every device
constexpr int kFrames = 4;

// A ring of per-frame slices in one buffer: each frame uploads only its
// slice and binds it as the kernel's whole input.
std::vector<vec4> runRing(Device &device) {
    auto ring = BufferResource<vec4>{kFrames * kFrameElements};
    auto results = BufferResource<vec4>{kFrames * kFrameElements};
    std::fill(results.begin(), results.end(), vec4{-1.0f});
    device.uploadBuffer(ring);
    device.uploadBuffer(results);
    auto kernel = tests::ScaleRange{};
    for (int frame = 0; frame < kFrames; ++frame) {
        auto const first = static_cast<std::size_t>(frame * kFrameElements);
        for (int i = 0; i < kFrameElements; ++i) {
            ring[static_cast<int>(first) + i] = vec4{static_cast<float>(frame), static_cast<float>(i), 1.0f, 0.5f};
        }
        device.uploadBuffer(ring, first, kFrameElements);
        kernel.in.attach(ring, first, kFrameElements);
        kernel.out.attach(results, first, kFrameElements);
        kernel.factor = 2.0f + static_cast<float>(frame);
        device.useKernel(kernel);
        device.bindBuffer(kernel.in);
        device.bindBuffer(kernel.out);
        device.setUniform(kernel.factor);
        device.execute(kernel, uvec3{1u, 1u, 1u});
    }
    device.downloadBuffer(results, kFrameElements, 2 * kFrameElements);
    return contents(results);
}

TEST_P(Equivalence, BufferRanges) {
    auto const [expected, actual] = runBoth<std::vector<vec4>>(runRing);
    for (int i = kFrameElements; i < 3 * kFrameElements; ++i) {
        EXPECT_EQ(expected[i], actual[i]) << i;
    }
    // frame 2, element 5: (2, 5, 1, 0.5) * 4 + 64.
    EXPECT_EQ(actual[2 * kFrameElements + 5], (vec4{72.0f, 84.0f, 68.0f, 66.0f}));
    // Not downloaded: the host copy keeps what it had.
    EXPECT_EQ(actual[0], vec4{-1.0f});
    EXPECT_EQ(actual.back(), vec4{-1.0f});
}

// GPU copies change only where a ranged upload says.
TEST_P(Equivalence, RangedUploadMovesOnlyTheRange) {
    auto source = BufferResource<vec4>{3 * kFrameElements};
    auto copy = BufferResource<vec4>{3 * kFrameElements};
    std::fill(source.begin(), source.end(), vec4{1.0f});
    gpu->uploadBuffer(source);
    gpu->uploadBuffer(copy);
    std::fill(source.begin(), source.end(), vec4{2.0f});
    gpu->uploadBuffer(source, kFrameElements, kFrameElements);
    auto kernel = tests::ScaleRange{};
    kernel.in.attach(source);
    kernel.out.attach(copy);
    kernel.factor = 1.0f;
    gpu->useKernel(kernel);
    gpu->bindBuffer(kernel.in);
    gpu->bindBuffer(kernel.out);
    gpu->setUniform(kernel.factor);
    gpu->execute(kernel, uvec3{3u, 1u, 1u});
    gpu->downloadBuffer(copy);
    auto const offset = static_cast<float>(3 * kFrameElements);
    EXPECT_EQ(copy[0], vec4{1.0f + offset});
    EXPECT_EQ(copy[kFrameElements], vec4{2.0f + offset});
    EXPECT_EQ(copy[2 * kFrameElements], vec4{1.0f + offset});

    auto const alignment = gpu->info().limits.storageBufferOffsetAlignment;
    if (alignment > sizeof(vec4)) {
        kernel.in.attach(source, 1, 8);
        EXPECT_THROW(gpu->bindBuffer(kernel.in), DeviceError);
    }
    kernel.in.attach(source, 0, 0);
    EXPECT_THROW(gpu->bindBuffer(kernel.in), DeviceError);
}

std::vector<vec4> copyBodies(Device &device, TableResource<tests::Body, Soa<16>> &table) {
    auto out = BufferResource<vec4>{static_cast<std::int32_t>(table.size())};
    auto kernel = tests::CopyBodies{};
    kernel.bodies.attach(table);
    kernel.out.attach(out);
    device.uploadBuffer(out);
    device.useKernel(kernel);
    device.bindTable(kernel.bodies);
    device.bindBuffer(kernel.out);
    device.execute(kernel, uvec3{1u, 1u, 1u});
    device.downloadBuffer(out);
    return contents(out);
}

tests::Body body(int const i, float const mass) {
    return tests::Body{vec3{static_cast<float>(i), 1.0f, 2.0f}, mass, static_cast<uint>(i)};
}

TEST_P(Equivalence, TableRanges) {
    constexpr int n = 50;
    auto table = TableResource<tests::Body, Soa<16>>{n};
    for (int i = 0; i < n; ++i) {
        table.set(i, body(i, 1.0f));
    }
    gpu->uploadTable(table);
    for (int i = 20; i < 26; ++i) {
        table.set(i, body(i, 5.0f));
    }
    table.set(40, body(40, 9.0f));
    gpu->uploadTable(table, 20, 6); // inside page 1 (elements 16..31)
    auto const seen = copyBodies(*gpu, table);
    EXPECT_EQ(seen[19].w, 1.0f + 19.0f);
    EXPECT_EQ(seen[20].w, 5.0f + 20.0f);
    EXPECT_EQ(seen[25].w, 5.0f + 25.0f);
    EXPECT_EQ(seen[26].w, 1.0f + 26.0f);
    EXPECT_EQ(seen[40].w, 1.0f + 40.0f) << "element 40 was not uploaded";

    // A range spanning two pages, then a ranged download.
    gpu->uploadTable(table, 30, 12);
    EXPECT_EQ(copyBodies(*gpu, table)[40].w, 9.0f + 40.0f);
    table.set(10, body(10, -3.0f));
    table.set(44, body(44, -3.0f));
    table.set(49, body(49, -3.0f));
    gpu->downloadTable(table, 40, 5);
    EXPECT_EQ(table.get(10).mass, -3.0f);
    EXPECT_EQ(table.get(49).mass, -3.0f) << "page 3, outside the downloaded words";
    EXPECT_EQ(table.get(44).mass, 1.0f) << "restored from the device";
    EXPECT_EQ(table.get(40).mass, 9.0f);
}

TEST(TableRanges, WordsOfCoverEveryFieldOfTheElements) {
    auto aos = TableResource<tests::Body>{10};
    auto const words = aos.wordsOf(2, 3);
    EXPECT_EQ(words.first, 4u + 2u * 5u);
    EXPECT_EQ(words.count, 3u * 5u);
    auto soa = TableResource<tests::Body, Soa<16>>{50};
    auto const range = soa.wordsOf(20, 6); // page 1, slots 4..9
    EXPECT_EQ(range.first, 4u + 16u * 5u + 4u * 3u);
    EXPECT_EQ(range.first + range.count, 4u + 16u * 5u + 16u * 4u + 10u);
    EXPECT_EQ(soa.wordsOf(0, 0).count, 0u);
}

// Medium precision: fp16-sized differences, and the declarations it adds.
TEST_P(Equivalence, MediumPrecision) {
    auto const [expected, actual] = runBoth<std::vector<vec4>>([](Device &device) {
        constexpr int n = 256;
        auto normals = BufferResource<vec4>{n};
        for (int i = 0; i < n; ++i) {
            auto const a = static_cast<float>(i) * 0.05f;
            normals[i] = vec4{std::cos(a), std::sin(a * 0.7f), 0.3f + 0.5f * std::sin(a), 1000.0f + static_cast<float>(i)};
        }
        auto shaded = BufferResource<vec4>{n};
        auto kernel = tests::MediumPrecisionShading{};
        kernel.normals.attach(normals);
        kernel.shaded.attach(shaded);
        kernel.light = vec3{0.3f, 0.8f, 0.5f};
        device.uploadBuffer(normals);
        device.uploadBuffer(shaded);
        device.useKernel(kernel);
        device.bindBuffer(kernel.normals);
        device.bindBuffer(kernel.shaded);
        device.setUniform(kernel.light);
        device.execute(kernel, uvec3{n / 64, 1u, 1u});
        device.downloadBuffer(shaded);
        return contents(shaded);
    });
    for (std::size_t i = 0; i < expected.size(); ++i) {
        for (int c = 0; c < 3; ++c) {
            // fp16 keeps about three decimal digits; pow and normalize lose more.
            EXPECT_NEAR(expected[i][c], actual[i][c], 1.5e-2f) << "element " << i << " component " << c;
        }
        EXPECT_EQ(expected[i].w, actual[i].w) << "copied at full precision";
    }
}

TEST(MediumPrecision, OnlyArithmeticIsRelaxed) {
    auto const program = ProgramRegistry::instance().find("medium_precision_shading");
    ASSERT_TRUE(program.has_value());
    auto const es = std::string{program->glslEs};
    auto const relaxed = es.find("precision mediump float;");
    ASSERT_NE(relaxed, std::string::npos);
    EXPECT_LT(es.find("buffer _normalsLayout"), relaxed) << "buffers are declared at full precision";
    EXPECT_LT(relaxed, es.find("vec3 tonemap(")) << "functions compute at medium precision";
    EXPECT_EQ(std::string{program->glslDesktop}.find("mediump"), std::string::npos);
    // SPIR-V: OpDecorate <id> RelaxedPrecision.
    auto decorations = 0;
    auto const words = program->spirv;
    for (std::size_t i = 5; i + 2 < words.size(); i += words[i] >> 16) {
        decorations += words[i] == ((3u << 16) | 71u) and words[i + 2] == 0u ? 1 : 0;
        if ((words[i] >> 16) == 0) {
            break;
        }
    }
    EXPECT_GT(decorations, 5);
}

TEST(KernelTemplates, EachInstanceIsAProgram) {
    auto const lambert = ProgramRegistry::instance().find("shade_lambert");
    auto const toon = ProgramRegistry::instance().find("shade_toon");
    ASSERT_TRUE(lambert.has_value());
    ASSERT_TRUE(toon.has_value());
    EXPECT_EQ(lambert->localSize, (uvec3{32u, 1u, 1u}));
    EXPECT_EQ(toon->localSize, (uvec3{64u, 1u, 1u}));
    using Toon64 = tests::Shade<tests::Toon, 64>;
    EXPECT_EQ(std::string_view{Toon64::fileLocation}, "shade_toon");
}

} // namespace
