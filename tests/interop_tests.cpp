// Sync files and platform frames written by kernels.

#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#include <unistd.h>

#include <external_frame/external_frame.hpp>
#include <tc/program.hpp>

#include "equivalence.hpp"
#include "kernels/dispatch_kernels.hpp"

#if defined(__ANDROID__)
#include <android/hardware_buffer.h>
#endif

namespace {

using namespace tc;
using namespace tc::testing_support;

TEST(SyncFence, SignalsWhenTheDescriptorBecomesReadable) {
    int ends[2] = {-1, -1};
    ASSERT_EQ(pipe(ends), 0);
    auto fence = SyncFence{ends[0]};
    EXPECT_FALSE(fence.isSignaled());
    char const byte = 1;
    ASSERT_EQ(write(ends[1], &byte, 1), 1);
    EXPECT_TRUE(fence.isSignaled());
    fence.wait();
    auto moved = std::move(fence);
    EXPECT_TRUE(fence.empty());
    EXPECT_EQ(moved.get(), ends[0]);
    close(ends[1]);
    EXPECT_TRUE(SyncFence{}.isSignaled());
}

// The fence of one submission, waited on by the host and by later work on
// the device, which then reads what the first submission wrote.
TEST_P(Equivalence, SyncFenceRoundTrip) {
    RecordProperty("sync_files", gpu->info().syncFiles ? "device" : "host");
    std::printf("[ info     ] %s '%s': sync files %s\n", gpu->info().backend.c_str(), gpu->info().deviceName.c_str(),
                gpu->info().syncFiles ? "on the device" : "waited on the host");
    for (auto *device : {cpu.get(), gpu.get()}) {
        auto values = BufferResource<vec4>{64};
        auto first = device->createBuffer<vec4>(64);
        auto second = BufferResource<vec4>{64};
        device->uploadBuffer(values);
        device->uploadBuffer(first);
        device->uploadBuffer(second);
        auto kernel = tests::ScaleRange{};
        device->useKernel(kernel);
        kernel.factor = 2.0f;
        device->setUniform(kernel.factor);
        kernel.in.attach(values);
        kernel.out.attach(first);
        device->bindBuffer(kernel.in);
        device->bindBuffer(kernel.out);
        device->execute(kernel, uvec3{1u, 1u, 1u});
        auto fence = device->submitWithFence();
        fence.wait();
        EXPECT_TRUE(fence.isSignaled());
        EXPECT_EQ(first[0], vec4{64.0f}) << device->info().backend << ": mapped results after the fence";

        device->execute(kernel, uvec3{1u, 1u, 1u});
        auto again = device->submitWithFence();
        device->waitFor(std::move(again)); // device-side wait where the API can import sync files
        kernel.in.attach(first);
        kernel.out.attach(second);
        device->bindBuffer(kernel.in);
        device->bindBuffer(kernel.out);
        device->execute(kernel, uvec3{1u, 1u, 1u});
        device->downloadBuffer(second);
        EXPECT_EQ(second[5], vec4{64.0f * 2.0f + 64.0f}) << device->info().backend;
        device->waitFor(SyncFence{});
    }
}

// A fence from one API, waited on by another: Vulkan and OpenGL ES work
// on the same GPU, ordered without a host wait.
TEST(SyncFence, CrossesApis) {
    auto request = DeviceRequest{};
    auto devices = std::vector<std::unique_ptr<Device>>{};
    for (auto const *backend : {"vulkan", "gles"}) {
        request.backend = backend;
        try {
            devices.push_back(createDevice(request));
        } catch (DeviceError const &error) {
            GTEST_SKIP() << error.what();
        }
        if (not devices.back()->info().syncFiles) {
            GTEST_SKIP() << backend << " cannot exchange sync files here";
        }
    }
    for (int from = 0; from < 2; ++from) {
        auto &producer = *devices[from];
        auto &consumer = *devices[1 - from];
        auto values = BufferResource<vec4>{64};
        auto out = BufferResource<vec4>{64};
        producer.uploadBuffer(values);
        producer.uploadBuffer(out);
        auto kernel = tests::ScaleRange{};
        kernel.in.attach(values);
        kernel.out.attach(out);
        kernel.factor = 1.0f;
        producer.useKernel(kernel);
        producer.bindBuffer(kernel.in);
        producer.bindBuffer(kernel.out);
        producer.setUniform(kernel.factor);
        producer.execute(kernel, uvec3{1u, 1u, 1u});
        auto fence = producer.submitWithFence();
        EXPECT_FALSE(fence.empty() and not fence.isSignaled());
        consumer.waitFor(std::move(fence));
        consumer.finish();
        producer.downloadBuffer(out);
        EXPECT_EQ(out[0], vec4{64.0f});
    }
}

TEST(ExternalImages, KernelsDeclareThemAsStorageImages) {
    auto const program = ProgramRegistry::instance().find("render_to_frame");
    ASSERT_TRUE(program.has_value());
    ASSERT_EQ(program->resources.size(), 1u);
    auto const &slot = program->resources[0];
    EXPECT_EQ(slot.kind, ResourceKind::ExternalImage);
    EXPECT_EQ(slot.access, Access::WriteOnly);
    EXPECT_EQ(glslFormatQualifier(slot.imageFormat), "rgba8");
    EXPECT_NE(program->glslEs.find("writeonly uniform highp image2D target"), std::string_view::npos);
}

#if defined(__ANDROID__)

using HardwareBuffer = std::unique_ptr<AHardwareBuffer, void (*)(AHardwareBuffer *)>;

HardwareBuffer allocatePresentable(ivec2 const size) {
    auto desc = AHardwareBuffer_Desc{};
    desc.width = static_cast<std::uint32_t>(size.x);
    desc.height = static_cast<std::uint32_t>(size.y);
    desc.layers = 1;
    desc.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
    desc.usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_GPU_FRAMEBUFFER
               | AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;
    AHardwareBuffer *buffer = nullptr;
    if (AHardwareBuffer_allocate(&desc, &buffer) != 0) {
        throw DeviceError{"AHardwareBuffer_allocate failed"};
    }
    return HardwareBuffer{buffer, AHardwareBuffer_release};
}

std::vector<std::uint8_t> readFrame(AHardwareBuffer *buffer, ivec2 const size) {
    auto desc = AHardwareBuffer_Desc{};
    AHardwareBuffer_describe(buffer, &desc);
    void *data = nullptr;
    EXPECT_EQ(AHardwareBuffer_lock(buffer, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, -1, nullptr, &data), 0);
    auto texels = std::vector<std::uint8_t>{};
    for (int y = 0; y < size.y; ++y) {
        auto const *row = static_cast<std::uint8_t const *>(data) + static_cast<std::size_t>(y) * desc.stride * 4;
        texels.insert(texels.end(), row, row + size.x * 4);
    }
    AHardwareBuffer_unlock(buffer, nullptr);
    return texels;
}

// Two frames of a presentation loop: render, hand over with a fence, get
// the frame back with a release fence, render again.
std::vector<std::uint8_t> renderFrames(Device &device, ivec2 const size) {
    auto const buffer = allocatePresentable(size);
    auto frame = ExternalImage::fromHardwareBuffer(buffer.get());
    auto kernel = examples::RenderToFrame{};
    kernel.target.attach(frame);
    device.useKernel(kernel);
    device.bindExternalImage(kernel.target);
    kernel.time = 0.25f;
    device.setUniform(kernel.time);
    auto const groups = uvec3{(static_cast<uint>(size.x) + 15u) / 16u, (static_cast<uint>(size.y) + 15u) / 16u, 1u};
    device.execute(kernel, groups);
    auto acquire = device.submitWithFence(); // what ASurfaceTransaction_setBuffer would get
    acquire.wait();                          // the compositor reads the frame...
    auto release = std::move(acquire);       // ...and gives it back with a release fence
    device.waitFor(std::move(release));
    kernel.time = 0.75f;
    device.setUniform(kernel.time);
    device.execute(kernel, groups);
    device.submitWithFence().wait();
    return readFrame(buffer.get(), size);
}

TEST_P(Equivalence, RenderIntoPlatformFrames) {
    auto const size = ivec2{96, 40};
    try {
        auto const [expected, actual] =
            runBoth<std::vector<std::uint8_t>>([size](Device &device) { return renderFrames(device, size); });
        ASSERT_EQ(expected.size(), actual.size());
        for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(expected[i], actual[i], 1) << "byte " << i;
        }
        // The bar of the second frame at x = 0.75, none at x = 0.25.
        auto const &texels = actual;
        auto const blue = [&texels, size](int const x, int const y) {
            return texels[(static_cast<std::size_t>(y) * size.x + x) * 4 + 2];
        };
        EXPECT_EQ(blue(72, 10), 255);
        EXPECT_EQ(blue(24, 10), 0);
    } catch (DeviceError const &error) {
        GTEST_SKIP() << error.what();
    }
}

#endif

} // namespace
