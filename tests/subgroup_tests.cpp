// Subgroup operations and 64-bit integers on every device that has them.

#include <algorithm>
#include <cstdint>
#include <vector>

#include "equivalence.hpp"
#include "kernels/subgroup_kernels.hpp"

namespace {

using namespace tc;
using namespace tc::testing_support;

struct SubgroupResult {
    std::vector<uint> prefix;
    std::vector<uint> sums;
    std::vector<uint> compacted; // sorted
    std::vector<uint> counters;
    std::vector<uint> checks;
    uint subgroupSize = 0;
};

constexpr int kGroups = 6;

SubgroupResult runSubgroupOps(Device &device) {
    constexpr int n = 64 * kGroups;
    auto values = BufferResource<uint>{n};
    auto seed = 3u;
    for (int i = 0; i < n; ++i) {
        seed = seed * 1664525u + 1013904223u;
        values[i] = seed >> 20;
    }
    auto prefix = BufferResource<uint>{n};
    auto sums = BufferResource<uint>{kGroups};
    auto compacted = BufferResource<uint>{n};
    auto counters = BufferResource<uint>{3};
    auto checks = BufferResource<uint>{n};
    auto kernel = tests::SubgroupOps{};
    kernel.values.attach(values);
    kernel.prefix.attach(prefix);
    kernel.sums.attach(sums);
    kernel.compacted.attach(compacted);
    kernel.counters.attach(counters);
    kernel.checks.attach(checks);
    for (auto *buffer : {&values, &prefix, &sums, &compacted, &counters, &checks}) {
        device.uploadBuffer(*buffer);
    }
    device.useKernel(kernel);
    device.bindBuffer(kernel.values);
    device.bindBuffer(kernel.prefix);
    device.bindBuffer(kernel.sums);
    device.bindBuffer(kernel.compacted);
    device.bindBuffer(kernel.counters);
    device.bindBuffer(kernel.checks);
    device.execute(kernel, uvec3{kGroups, 1u, 1u});
    for (auto *buffer : {&prefix, &sums, &compacted, &counters, &checks}) {
        device.downloadBuffer(*buffer);
    }
    auto kept = std::vector<uint>(compacted.begin(), compacted.begin() + counters[0]);
    std::sort(kept.begin(), kept.end());
    return SubgroupResult{contents(prefix), contents(sums),  kept, contents(counters), contents(checks),
                          device.info().limits.subgroupSize};
}

TEST_P(Equivalence, SubgroupOperations) {
    try {
        auto const [expected, actual] = runBoth<SubgroupResult>(runSubgroupOps);
        RecordProperty("subgroup_size", static_cast<int>(actual.subgroupSize));
        EXPECT_EQ(expected.prefix, actual.prefix);
        EXPECT_EQ(expected.sums, actual.sums);
        EXPECT_EQ(expected.compacted, actual.compacted);
        EXPECT_EQ(expected.counters[0], actual.counters[0]);
        EXPECT_EQ(expected.counters[1], actual.counters[1]);
        for (std::size_t i = 0; i < actual.checks.size(); ++i) {
            EXPECT_EQ(actual.checks[i], 0x3fffu) << "invocation " << i;
            EXPECT_EQ(expected.checks[i], 0x3fffu) << "CPU invocation " << i;
        }
        // One elected invocation per subgroup.
        auto const subgroupsPerGroup = (64u + actual.subgroupSize - 1u) / actual.subgroupSize;
        EXPECT_EQ(actual.counters[2], kGroups * subgroupsPerGroup) << "subgroup size " << actual.subgroupSize;
        EXPECT_EQ(actual.prefix[63], actual.sums[0]);
    } catch (SkipKernel const &skip) {
        GTEST_SKIP() << skip.reason;
    }
}

// Other subgroup sizes on the CPU: the results stay the same.
TEST(CpuSubgroups, ResultsDoNotDependOnTheSubgroupSize) {
    auto reference = std::optional<SubgroupResult>{};
    for (std::uint32_t size : {1u, 4u, 16u, 32u, 64u, 128u}) {
        auto device = cpu::createDevice(cpu::CpuDeviceConfig{2, nullptr, size});
        auto const result = runSubgroupOps(*device);
        EXPECT_EQ(result.subgroupSize, size);
        for (auto const check : result.checks) {
            ASSERT_EQ(check, 0x3fffu) << "subgroup size " << size;
        }
        EXPECT_EQ(result.counters[2], kGroups * ((64u + size - 1u) / size));
        if (reference) {
            EXPECT_EQ(result.prefix, reference->prefix) << size;
            EXPECT_EQ(result.compacted, reference->compacted) << size;
        } else {
            reference = result;
        }
    }
    EXPECT_THROW(cpu::createDevice(cpu::CpuDeviceConfig{1, nullptr, 24}), DeviceError);
}

TEST_P(Equivalence, Integers64) {
    try {
        auto const [expected, actual] = runBoth<std::vector<i64>>([](Device &device) {
            constexpr int n = 128;
            auto keys = BufferResource<u64>{n};
            for (int i = 0; i < n; ++i) {
                keys[i] = 0x0123456789abcdefull * static_cast<u64>(i + 1);
            }
            auto out = BufferResource<i64>{2 * n};
            auto kernel = tests::Wide64{};
            kernel.keys.attach(keys);
            kernel.out.attach(out);
            device.uploadBuffer(keys);
            device.uploadBuffer(out);
            device.useKernel(kernel);
            device.bindBuffer(kernel.keys);
            device.bindBuffer(kernel.out);
            device.execute(kernel, uvec3{2u, 1u, 1u});
            device.downloadBuffer(out);
            return contents(out);
        });
        EXPECT_EQ(expected, actual);
        EXPECT_EQ(actual[2 * 5 + 1] >> 40, 5);
    } catch (SkipKernel const &skip) {
        GTEST_SKIP() << skip.reason;
    }
}

TEST_P(Equivalence, Atomics64) {
    try {
        auto const [expected, actual] = runBoth<std::vector<u64>>([](Device &device) {
            auto pixels = BufferResource<u64>{97};
            auto kernel = tests::VisibilityBuffer{};
            kernel.pixels.attach(pixels);
            device.uploadBuffer(pixels);
            device.useKernel(kernel);
            device.bindBuffer(kernel.pixels);
            device.execute(kernel, uvec3{40u, 1u, 1u});
            device.downloadBuffer(pixels);
            return contents(pixels);
        });
        EXPECT_EQ(expected, actual);
        EXPECT_NE(actual[0], 0u);
    } catch (SkipKernel const &skip) {
        GTEST_SKIP() << skip.reason;
    }
}

} // namespace
