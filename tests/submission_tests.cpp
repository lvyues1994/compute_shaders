// Asynchronous submission and the barriers between dependent dispatches.

#include <array>
#include <thread>
#include <vector>

#include "equivalence.hpp"
#include "kernels/dispatch_kernels.hpp"

namespace {

using namespace tc;
using namespace tc::testing_support;

constexpr int kSlice = 64;  // vec4 elements: 1 KiB, aligned for every device
constexpr int kInFlight = 3;
constexpr int kFrames = 10;

// The engine's frame loop: frame f fills slot f % 3 of a ring in mapped
// memory, the GPU scales it into the same slot of a mapped results ring,
// and the host reads frame f's results once frame f's submission is done,
// while later frames are being recorded and run.
std::vector<vec4> pipelinedFrames(Device &device) {
    auto inputs = device.createBuffer<vec4>(kInFlight * kSlice);
    auto results = device.createBuffer<vec4>(kInFlight * kSlice);
    device.uploadBuffer(inputs);
    device.uploadBuffer(results);
    auto tickets = std::array<Submission, kInFlight>{};
    auto frameOfSlot = std::array<int, kInFlight>{-1, -1, -1};
    auto seen = std::vector<vec4>(kFrames);
    auto const collect = [&](int const slot) {
        tickets[slot].wait();
        if (frameOfSlot[slot] >= 0) {
            seen[frameOfSlot[slot]] = results[slot * kSlice + 7];
        }
    };
    auto kernel = tests::ScaleRange{};
    device.useKernel(kernel);
    for (int frame = 0; frame < kFrames; ++frame) {
        auto const slot = frame % kInFlight;
        collect(slot); // the slot's previous frame is done: its memory is free
        auto const first = static_cast<std::size_t>(slot * kSlice);
        for (int i = 0; i < kSlice; ++i) {
            inputs[slot * kSlice + i] = vec4{static_cast<float>(frame), static_cast<float>(i), 0.0f, 1.0f};
        }
        kernel.in.attach(inputs, first, kSlice);
        kernel.out.attach(results, first, kSlice);
        kernel.factor = 2.0f;
        device.bindBuffer(kernel.in);
        device.bindBuffer(kernel.out);
        device.setUniform(kernel.factor);
        device.execute(kernel, uvec3{1u, 1u, 1u});
        tickets[slot] = device.submit();
        frameOfSlot[slot] = frame;
    }
    for (int slot = 0; slot < kInFlight; ++slot) {
        collect(slot);
    }
    return seen;
}

TEST_P(Equivalence, FramesInFlight) {
    auto const [expected, actual] = runBoth<std::vector<vec4>>(pipelinedFrames);
    EXPECT_EQ(expected, actual);
    for (int frame = 0; frame < kFrames; ++frame) {
        EXPECT_EQ(actual[frame], (vec4{2.0f * frame + 64.0f, 14.0f + 64.0f, 64.0f, 66.0f})) << "frame " << frame;
    }
}

// Ping-pong needs a barrier every time; the independent chains in between
// must not lose their order either.
std::vector<vec4> dependentChains(Device &device) {
    auto a = BufferResource<vec4>{kSlice};
    auto b = BufferResource<vec4>{kSlice};
    auto shared = BufferResource<vec4>{2 * kSlice};
    for (int i = 0; i < kSlice; ++i) {
        a[i] = vec4{1.0f};
        shared[i] = vec4{static_cast<float>(i)};
    }
    for (auto *buffer : {&a, &b}) {
        device.uploadBuffer(*buffer);
    }
    device.uploadBuffer(shared);
    auto kernel = tests::ScaleRange{};
    device.useKernel(kernel);
    kernel.factor = 0.5f;
    device.setUniform(kernel.factor);
    for (int step = 0; step < 8; ++step) {
        kernel.in.attach(step % 2 == 0 ? a : b);
        kernel.out.attach(step % 2 == 0 ? b : a);
        device.bindBuffer(kernel.in);
        device.bindBuffer(kernel.out);
        device.execute(kernel, uvec3{1u, 1u, 1u});
        // The two halves of `shared`: the upper half reads what the lower
        // half's dispatch just wrote.
        kernel.in.attach(shared, 0, kSlice);
        kernel.out.attach(shared, kSlice, kSlice);
        device.bindBuffer(kernel.in);
        device.bindBuffer(kernel.out);
        device.execute(kernel, uvec3{1u, 1u, 1u});
        kernel.in.attach(shared, kSlice, kSlice);
        kernel.out.attach(shared, 0, kSlice);
        device.bindBuffer(kernel.in);
        device.bindBuffer(kernel.out);
        device.execute(kernel, uvec3{1u, 1u, 1u});
        if (step == 3) {
            device.submit(); // the dependency also holds across submissions
        }
    }
    device.downloadBuffer(a);
    device.downloadBuffer(shared);
    auto result = contents(a);
    result.insert(result.end(), shared.begin(), shared.end());
    return result;
}

TEST_P(Equivalence, DependentDispatchesKeepTheirOrder) {
    auto const [expected, actual] = runBoth<std::vector<vec4>>(dependentChains);
    EXPECT_EQ(expected, actual);
    // a after 8 steps of x * 0.5 + 64 from 1: converges to 128.
    EXPECT_NEAR(actual[0].x, 128.0f - 127.0f / 256.0f, 1e-3f);
}

TEST_P(Equivalence, SubmissionsCanBeAwaitedAnywhere) {
    auto values = BufferResource<vec4>{kSlice};
    auto out = gpu->createBuffer<vec4>(kSlice);
    gpu->uploadBuffer(values);
    gpu->uploadBuffer(out);
    auto kernel = tests::ScaleRange{};
    kernel.in.attach(values);
    kernel.out.attach(out);
    kernel.factor = 3.0f;
    gpu->useKernel(kernel);
    gpu->bindBuffer(kernel.in);
    gpu->bindBuffer(kernel.out);
    gpu->setUniform(kernel.factor);
    gpu->execute(kernel, uvec3{1u, 1u, 1u});
    auto const ticket = gpu->submit();
    auto const nothingNew = gpu->submit();
    auto waiter = std::thread{[ticket] { ticket.wait(); }};
    waiter.join();
    EXPECT_TRUE(ticket.isComplete());
    nothingNew.wait();
    EXPECT_TRUE(nothingNew.isComplete());
    EXPECT_EQ(out[3], vec4{64.0f}) << "mapped memory holds the results once the submission is done";

    gpu->execute(kernel, uvec3{1u, 1u, 1u});
    auto const last = gpu->submit();
    gpu.reset(); // the device waits for its work; the ticket outlives it
    last.wait();
    EXPECT_TRUE(last.isComplete());
}

TEST(Submission, DefaultIsComplete) {
    auto const done = Submission{};
    EXPECT_TRUE(done.isComplete());
    done.wait();
}

} // namespace
