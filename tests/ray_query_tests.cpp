// Ray queries: the CPU traversal and the GPU's agree on what each ray hits.

#include <cmath>
#include <string>
#include <vector>

#include "equivalence.hpp"
#include "kernels/ray_query_kernels.hpp"

namespace {

using namespace tc;
using namespace tc::testing_support;

// A unit quad (two opaque triangles, counter-clockwise seen from +z) and,
// as a second geometry, two triangles that are not opaque, at z 0.5 and 0.75.
struct Scene {
    BufferResource<vec4> positions{10};
    BufferResource<uint> indices{12};

    Scene() {
        vec4 const corners[] = {{-1.0f, -1.0f, 0.0f, 1.0f},  {1.0f, -1.0f, 0.0f, 1.0f},  {1.0f, 1.0f, 0.0f, 1.0f},
                                {-1.0f, 1.0f, 0.0f, 1.0f},   {-0.5f, -0.5f, 0.5f, 1.0f}, {0.5f, -0.5f, 0.5f, 1.0f},
                                {0.0f, 0.5f, 0.5f, 1.0f},    {-0.5f, -0.5f, 0.75f, 1.0f}, {0.5f, -0.5f, 0.75f, 1.0f},
                                {0.0f, 0.5f, 0.75f, 1.0f}};
        uint const triangles[] = {0, 1, 2, 0, 2, 3, 0, 1, 2, 3, 4, 5};
        for (int i = 0; i < 10; ++i) {
            positions[i] = corners[i];
        }
        for (int i = 0; i < 12; ++i) {
            indices[i] = triangles[i];
        }
    }
};

struct Ray {
    vec3 origin;
    vec3 direction;
    float tMax;
};

// Through the non-opaque triangles onto the quad, onto either half of the
// quad, onto the mirrored instance, from below, past everything, short of
// the quad, at a slant, and from between instances 0 and 2.
std::vector<Ray> const kRays = {
    {{0.3f, -0.45f, 5.0f}, {0.0f, 0.0f, -1.0f}, 100.0f}, {{0.8f, 0.6f, 5.0f}, {0.0f, 0.0f, -1.0f}, 100.0f},
    {{-0.7f, 0.2f, 5.0f}, {0.0f, 0.0f, -1.0f}, 100.0f},  {{3.4f, 0.25f, 5.0f}, {0.0f, 0.0f, -1.0f}, 100.0f},
    {{0.3f, -0.45f, -5.0f}, {0.0f, 0.0f, 1.0f}, 100.0f}, {{10.0f, 10.0f, 5.0f}, {0.0f, 0.0f, -1.0f}, 100.0f},
    {{0.8f, 0.6f, 5.0f}, {0.0f, 0.0f, -1.0f}, 3.0f},     {{-2.0f, 0.1f, 3.0f}, {0.5f, 0.05f, -1.0f}, 100.0f},
    {{0.8f, 0.6f, -1.0f}, {0.0f, 0.0f, -1.0f}, 100.0f},
};

struct Trace {
    std::string what;
    uint flags = gl_RayFlagsNoneEXT;
    uint mask = 0xFFu;
};

std::vector<Trace> const kTraces = {
    {"no flags"},
    {"opaque", gl_RayFlagsOpaqueEXT},
    {"cull back faces", gl_RayFlagsCullBackFacingTrianglesEXT},
    {"terminate on first hit", gl_RayFlagsTerminateOnFirstHitEXT},
    {"cull mask 1", gl_RayFlagsNoneEXT, 0x01u},
    {"cull non-opaque", gl_RayFlagsCullNoOpaqueEXT},
};

struct Traced {
    std::vector<vec4> hits;
    std::vector<uvec4> ids;
};

std::vector<Traced> traceScene(Device &device) {
    auto scene = Scene{};
    device.uploadBuffer(scene.positions);
    device.uploadBuffer(scene.indices);
    auto const blas = device.createBottomLevel({TriangleGeometry::of(scene.positions, 0, 4, scene.indices, 0, 2, true),
                                                TriangleGeometry::of(scene.positions, 4, 6, scene.indices, 6, 2, false)});
    EXPECT_EQ(blas.triangleCount(), 4u);

    // Instance 0 as is, 1 mirrored along x and moved by 3, 2 behind 0.
    auto offsets = BufferResource<vec4>{3};
    offsets[0] = vec4{0.0f, 0.0f, 0.0f, 1.0f};
    offsets[1] = vec4{3.0f, 0.0f, 0.0f, -1.0f};
    offsets[2] = vec4{0.0f, 0.0f, -2.0f, 1.0f};
    device.uploadBuffer(offsets);
    auto instances = device.createDeviceBuffer<TopLevelInstance>(3);
    auto write = tests::WriteInstances{};
    write.offsets.attach(offsets);
    write.instances.attach(instances);
    write.blas = blas.reference();
    device.useKernel(write);
    device.bindBuffer(write.offsets);
    device.bindBuffer(write.instances);
    device.setUniform(write.blas);
    device.execute(write, uvec3{1u, 1u, 1u});
    auto const tlas = device.createTopLevel(4);
    device.buildTopLevel(tlas, instances, 0, 3);

    auto const count = static_cast<int>(kRays.size());
    auto rays = BufferResource<vec4>{2 * count};
    for (int i = 0; i < count; ++i) {
        rays[2 * i] = vec4{kRays[i].origin, kRays[i].tMax};
        rays[2 * i + 1] = vec4{kRays[i].direction, 0.0f};
    }
    auto hits = BufferResource<vec4>{count};
    auto ids = BufferResource<uvec4>{count};
    for (auto *buffer : {&rays, &hits}) {
        device.uploadBuffer(*buffer);
    }
    device.uploadBuffer(ids);
    auto trace = tests::TraceRays{};
    trace.scene.attach(tlas);
    trace.rays.attach(rays);
    trace.hits.attach(hits);
    trace.ids.attach(ids);
    auto results = std::vector<Traced>{};
    auto const run = [&](uint const flags, uint const mask) {
        trace.flags = flags;
        trace.cullMask = mask;
        device.useKernel(trace);
        device.bindAccelerationStructure(trace.scene);
        device.bindBuffer(trace.rays);
        device.bindBuffer(trace.hits);
        device.bindBuffer(trace.ids);
        device.setUniform(trace.flags);
        device.setUniform(trace.cullMask);
        device.execute(trace, uvec3{1u, 1u, 1u});
        device.downloadBuffer(hits);
        device.downloadBuffer(ids);
        results.push_back(Traced{contents(hits), contents(ids)});
    };
    for (auto const &config : kTraces) {
        run(config.flags, config.mask);
    }
    device.buildTopLevel(tlas, instances, 0, 1); // only instance 0 is left
    run(gl_RayFlagsNoneEXT, 0xFFu);
    return results;
}

TEST_P(Equivalence, RayQueriesHitTheSameTriangles) {
    if (not uses(gpu->info().limits.features, KernelFeature::RayQuery)) {
        EXPECT_THROW(gpu->createTopLevel(1), DeviceError) << gpu->info().backend;
        GTEST_SKIP() << gpu->info().deviceName << " (" << gpu->info().backend << ") has no ray queries";
    }
    auto const [expected, actual] = runBoth<std::vector<Traced>>(traceScene);
    ASSERT_EQ(expected.size(), actual.size());
    for (std::size_t trace = 0; trace < expected.size(); ++trace) {
        auto const what = trace < kTraces.size() ? kTraces[trace].what : std::string{"one instance"};
        auto const anyHit = trace < kTraces.size() and (kTraces[trace].flags & gl_RayFlagsTerminateOnFirstHitEXT) != 0u;
        for (std::size_t ray = 0; ray < kRays.size(); ++ray) {
            auto const &e = expected[trace];
            auto const &a = actual[trace];
            auto const where = what + ", ray " + std::to_string(ray);
            ASSERT_EQ(e.ids[ray].x == 0xFFFFFFFFu, a.ids[ray].x == 0xFFFFFFFFu) << where << ": hit or miss";
            if (anyHit or e.ids[ray].x == 0xFFFFFFFFu) {
                continue;
            }
            EXPECT_EQ(e.ids[ray], a.ids[ray]) << where << ": (custom index, instance, primitive, geometry)";
            EXPECT_NEAR(e.hits[ray].x, a.hits[ray].x, 1e-4f * std::abs(e.hits[ray].x)) << where << ": t";
            EXPECT_NEAR(e.hits[ray].y, a.hits[ray].y, 1e-4f) << where << ": barycentric u";
            EXPECT_NEAR(e.hits[ray].z, a.hits[ray].z, 1e-4f) << where << ": barycentric v";
            EXPECT_EQ(e.hits[ray].w, a.hits[ray].w) << where << ": front face";
        }
    }
    // The non-opaque triangle at z 0.75 (odd: rejected) lets the one at 0.5 through.
    EXPECT_EQ(actual[0].ids[0], (uvec4{100u, 0u, 0u, 1u}));
    EXPECT_NEAR(actual[0].hits[0].x, 4.5f, 1e-5f);
    EXPECT_EQ(actual[1].ids[0], (uvec4{100u, 0u, 1u, 1u})) << "opaque: the first triangle counts";
    EXPECT_EQ(actual[0].ids[3], (uvec4{101u, 1u, 1u, 0u})) << "the mirrored instance";
    EXPECT_EQ(actual[0].ids[4].x, 102u) << "from below, instance 2 first";
    EXPECT_EQ(actual[0].ids[5].x, 0xFFFFFFFFu);
    EXPECT_EQ(actual[0].ids[6].x, 0xFFFFFFFFu) << "tMax ends the ray before the quad";
    EXPECT_EQ(actual[4].ids[3].x, 0xFFFFFFFFu) << "the cull mask skips instance 1";
    EXPECT_EQ(actual[5].ids[0], (uvec4{100u, 0u, 0u, 0u})) << "without the non-opaque triangles";
    EXPECT_EQ(actual[6].ids[3].x, 0xFFFFFFFFu) << "rebuilt with one instance";
}

TEST(RayQuery, TopLevelsNeedLiveBottomLevels) {
    auto device = cpu::createDevice();
    auto instances = BufferResource<TopLevelInstance>{1};
    instances[0] = makeInstance(mat4{1.0f}, uvec2{12345u, 0u}, 0u);
    auto const tlas = device->createTopLevel(1);
    EXPECT_THROW(device->buildTopLevel(tlas, instances, 0, 1), DeviceError);
    EXPECT_THROW(device->buildTopLevel(tlas, instances, 0, 2), DeviceError) << "past the end";
    EXPECT_THROW(device->createTopLevel(0), DeviceError);
    EXPECT_THROW(device->createBottomLevel({}), DeviceError);
    auto scene = Scene{};
    EXPECT_THROW(device->createBottomLevel({TriangleGeometry::of(scene.positions, 0, 4, scene.indices, 0, 5)}), DeviceError)
        << "indices past the end";
    EXPECT_THROW(device->createBottomLevel({TriangleGeometry::of(scene.positions, 0, 2, scene.indices, 0, 2)}), DeviceError)
        << "an index past the vertices";
}

} // namespace
