// The CPU backend's parallel executor: injected executors, tc's own pool,
// and dispatches issued from inside pool tasks.

#include <atomic>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <tc/cpu/cpu_device.hpp>
#include <tc/tc.hpp>

namespace {

using namespace tc;

// Sums each workgroup's values through Shared memory and barriers.
struct TC_KERNEL GroupSums {
    static constexpr char fileLocation[] = "group_sums";
    static constexpr bool usesBarrier = true;
    uvec3 local_size{16, 1, 1};

    BufferBinding<uint, 0> values;
    BufferBinding<uint, 1> sums;
    Shared<std::array<uint, 16>> partial;

    void main() {
        uint const local = gl_LocalInvocationIndex;
        partial[local] = values[gl_GlobalInvocationID.x];
        barrier();
        for (uint stride = 8u; stride > 0u; stride /= 2u) {
            if (local < stride) {
                partial[local] += partial[local + stride];
            }
            barrier();
        }
        if (local == 0u) {
            sums[gl_WorkGroupID.x] = partial[0];
        }
    }
};

// Runs GroupSums over `groups` workgroups and checks every sum.
void sumGroups(Device &device, uint const groups) {
    auto values = BufferResource<uint>{static_cast<std::int32_t>(groups * 16u)};
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[static_cast<std::int32_t>(i)] = static_cast<uint>(i);
    }
    auto sums = BufferResource<uint>{static_cast<std::int32_t>(groups)};
    auto kernel = GroupSums{};
    kernel.values.attach(values);
    kernel.sums.attach(sums);
    device.useKernel(kernel);
    device.bindBuffer(kernel.values);
    device.bindBuffer(kernel.sums);
    device.execute(kernel, uvec3{groups, 1u, 1u});
    for (uint g = 0; g < groups; ++g) {
        EXPECT_EQ(sums[static_cast<std::int32_t>(g)], 16u * (16u * g) + 120u) << "group " << g;
    }
}

// Forwards to another executor and records the threads tasks ran on.
struct RecordingExecutor final : cpu::ParallelExecutor {
    explicit RecordingExecutor(std::shared_ptr<cpu::ParallelExecutor> inner_) : inner(std::move(inner_)) {}

    std::uint32_t concurrency() const override { return inner->concurrency(); }

    void bulk(std::uint32_t const count, cpu::TaskRef const task) override {
        ++calls;
        inner->bulk(count, [&](std::uint32_t const i) {
            {
                auto const lock = std::lock_guard<std::mutex>{mutex};
                threads.insert(std::this_thread::get_id());
            }
            task(i);
        });
    }

    std::shared_ptr<cpu::ParallelExecutor> inner;
    std::atomic<int> calls{0};
    std::mutex mutex;
    std::set<std::thread::id> threads;
};

TEST(CpuExecutor, InjectedExecutorRunsTheWorkgroups) {
    auto executor = std::make_shared<RecordingExecutor>(cpu::makeThreadPool(4));
    auto device = cpu::createDevice(cpu::CpuDeviceConfig{0, executor});
    EXPECT_EQ(device->info().deviceName, "host (4 threads)");
    sumGroups(*device, 64);
    sumGroups(*device, 3);
    EXPECT_EQ(executor->calls.load(), 2);
    // The pool's three workers and this thread, nothing created per dispatch.
    EXPECT_LE(executor->threads.size(), 4u);
    auto const firstThreads = executor->threads;
    sumGroups(*device, 64);
    for (auto const &id : executor->threads) {
        EXPECT_EQ(firstThreads.count(id), 1u) << "a dispatch ran on a new thread";
    }
}

// Devices on one pool, each dispatching from inside a task of that pool:
// the caller joins its own bulk(), so a fully busy pool still finishes.
TEST(CpuExecutor, DispatchesFromPoolTasksComplete) {
    auto pool = cpu::makeThreadPool(2);
    auto finished = std::atomic<int>{0};
    pool->bulk(6, [&](std::uint32_t) {
        auto device = cpu::createDevice(cpu::CpuDeviceConfig{0, pool});
        sumGroups(*device, 20);
        ++finished;
    });
    EXPECT_EQ(finished.load(), 6);
}

TEST(CpuExecutor, SingleThreadPoolRunsOnTheCaller) {
    auto executor = std::make_shared<RecordingExecutor>(cpu::makeThreadPool(1));
    auto device = cpu::createDevice(cpu::CpuDeviceConfig{0, executor});
    sumGroups(*device, 10);
    ASSERT_EQ(executor->threads.size(), 1u);
    EXPECT_EQ(*executor->threads.begin(), std::this_thread::get_id());
}

TEST(CpuExecutor, BulkRunsEveryIndexOnce) {
    auto pool = cpu::makeThreadPool(8);
    auto hits = std::vector<std::atomic<int>>(1000);
    pool->bulk(1000, [&](std::uint32_t const i) { ++hits[i]; });
    for (auto const &hit : hits) {
        EXPECT_EQ(hit.load(), 1);
    }
}

} // namespace
