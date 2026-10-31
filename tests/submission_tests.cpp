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

// Devices made and dropped beside this one, on its display or passing over it to find
// another, leave it working.
TEST_P(Equivalence, OtherDevicesComeAndGo) {
    for (auto const *const name : {"", "llvmpipe"}) {
        auto request = DeviceRequest{};
        request.backend = GetParam().backend;
        request.deviceNameContains = name;
        try {
            createDevice(request).reset();
        } catch (DeviceError const &) {
        }
    }
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
    gpu->submitWithFence().submission.wait();
    EXPECT_EQ(out[3], vec4{64.0f});
}

TEST(Submission, DefaultIsComplete) {
    auto const done = Submission{};
    EXPECT_TRUE(done.isComplete());
    done.wait();
    EXPECT_EQ(done.sequence(), 0u);
}

// One device, one dispatch of ScaleRange over `values`.
struct TimedScale {
    explicit TimedScale(Device &device_) : device(device_), values(device_.createBuffer<vec4>(kSlice)) {
        device.uploadBuffer(values);
        kernel.in.attach(values);
        kernel.out.attach(values);
        kernel.factor = 1.0f;
        device.useKernel(kernel);
        device.bindBuffer(kernel.in);
        device.bindBuffer(kernel.out);
        device.setUniform(kernel.factor);
    }
    void execute(uint const groups) { device.execute(kernel, uvec3{groups, 1u, 1u}); }

    Device &device;
    BufferResource<vec4> values;
    tests::ScaleRange kernel;
};

// Each call takes the timings of one submission (and earlier ones not
// taken), never those of later or unsubmitted work.
TEST_P(Equivalence, TimingsPerSubmission) {
    for (auto *device : {cpu.get(), gpu.get()}) {
        if (not device->info().timers) {
            continue;
        }
        auto const name = device->info().backend;
        auto scale = TimedScale{*device};
        device->enableTimers(true);
        scale.execute(1);
        auto const first = device->submit();
        scale.execute(2);
        scale.execute(3);
        auto const second = device->submit();
        scale.execute(4); // recorded, not submitted
        EXPECT_EQ(second.sequence(), first.sequence() + 1) << name;

        auto const one = device->takeTimings(first);
        ASSERT_EQ(one.size(), 1u) << name;
        EXPECT_EQ(one[0].workGroups, (uvec3{1u, 1u, 1u})) << name;
        EXPECT_EQ(one[0].submission, first.sequence()) << name;
        EXPECT_TRUE(device->takeTimings(first).empty()) << name << ": taken already";

        auto const two = device->takeTimings(second);
        ASSERT_EQ(two.size(), 2u) << name;
        EXPECT_EQ(two[0].workGroups.x, 2u) << name;
        EXPECT_EQ(two[1].workGroups.x, 3u) << name;
        EXPECT_EQ(two[1].submission, second.sequence()) << name;

        auto const fenced = device->submitWithFence();
        fenced.fence.wait();
        EXPECT_EQ(fenced.submission.sequence(), second.sequence() + 1) << name;
        auto const three = device->takeTimings(fenced.submission);
        ASSERT_EQ(three.size(), 1u) << name << ": the dispatch recorded before submitWithFence";
        EXPECT_EQ(three[0].workGroups.x, 4u) << name;

        scale.execute(5);
        auto const a = device->submit();
        scale.execute(6);
        auto const b = device->submit();
        auto const both = device->takeTimings(b);
        ASSERT_EQ(both.size(), 2u) << name << ": a later submission includes earlier ones";
        EXPECT_EQ(both[0].submission, a.sequence()) << name;
        EXPECT_TRUE(device->takeTimings(a).empty()) << name;

        scale.execute(7);
        EXPECT_EQ(device->takeTimings().size(), 1u) << name << ": takeTimings() still takes unsubmitted work";
        device->enableTimers(false);
    }
}

// The engine's frame loop with timers: each frame's timings are taken when
// its slot is reused, while the next frames are in flight.
TEST_P(Equivalence, TimedFramesInFlight) {
    for (auto *device : {cpu.get(), gpu.get()}) {
        if (not device->info().timers) {
            continue;
        }
        auto const name = device->info().backend;
        auto scale = TimedScale{*device};
        device->enableTimers(true);
        auto tickets = std::array<Submission, kInFlight>{};
        auto taken = 0;
        for (int frame = 0; frame < kFrames + kInFlight; ++frame) {
            auto const slot = frame % kInFlight;
            auto const timings = device->takeTimings(tickets[slot]);
            if (frame >= kInFlight) {
                ASSERT_EQ(timings.size(), 2u) << name << " frame " << frame - kInFlight;
                EXPECT_EQ(timings[0].workGroups.x, static_cast<uint>(frame - kInFlight + 1)) << name;
                EXPECT_EQ(timings[1].submission, tickets[slot].sequence()) << name;
                taken += 2;
            } else {
                EXPECT_TRUE(timings.empty()) << name;
            }
            if (frame < kFrames) {
                scale.execute(static_cast<uint>(frame + 1));
                scale.execute(1);
                tickets[slot] = device->submit();
            } else {
                tickets[slot] = Submission{};
            }
        }
        EXPECT_EQ(taken, 2 * kFrames) << name;
        EXPECT_TRUE(device->takeTimings().empty()) << name;
        device->enableTimers(false);
    }
}

} // namespace
