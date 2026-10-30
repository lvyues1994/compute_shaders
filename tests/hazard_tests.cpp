// Where the GPU backends put memory barriers.

#include <gtest/gtest.h>

#include <tc/detail/hazards.hpp>

namespace {

using tc::detail::HazardTracker;
using tc::detail::ResourceAccess;

int a = 0;
int b = 0;

ResourceAccess reads(void const *key, std::size_t const offset = 0, std::size_t const size = 0) {
    return ResourceAccess{key, offset, size, true, false};
}

ResourceAccess writes(void const *key, std::size_t const offset = 0, std::size_t const size = 0) {
    return ResourceAccess{key, offset, size, false, true};
}

TEST(HazardTracker, IndependentCommandsNeedNoBarrier) {
    auto hazards = HazardTracker{};
    EXPECT_FALSE(hazards.needsBarrier({writes(&a)}));
    EXPECT_FALSE(hazards.needsBarrier({writes(&b)}));
    auto readers = HazardTracker{};
    EXPECT_FALSE(readers.needsBarrier({reads(&a)}));
    EXPECT_FALSE(readers.needsBarrier({reads(&a), reads(&b)})) << "reads never conflict with reads";
}

TEST(HazardTracker, ReadAfterWrite) {
    auto hazards = HazardTracker{};
    EXPECT_FALSE(hazards.needsBarrier({writes(&a)}));
    EXPECT_TRUE(hazards.needsBarrier({reads(&a)}));
    EXPECT_FALSE(hazards.needsBarrier({reads(&a)})) << "two reads after the barrier";
}

TEST(HazardTracker, WriteAfterReadAndWrite) {
    auto hazards = HazardTracker{};
    EXPECT_FALSE(hazards.needsBarrier({reads(&a), writes(&b)}));
    EXPECT_TRUE(hazards.needsBarrier({writes(&a)})) << "write after read";
    EXPECT_TRUE(hazards.needsBarrier({writes(&a)})) << "write after write";
}

TEST(HazardTracker, RangesOfOneBuffer) {
    auto hazards = HazardTracker{};
    EXPECT_FALSE(hazards.needsBarrier({writes(&a, 0, 1024)}));
    EXPECT_FALSE(hazards.needsBarrier({writes(&a, 1024, 1024)})) << "ring buffer slices";
    EXPECT_FALSE(hazards.needsBarrier({reads(&a, 2048, 16)}));
    EXPECT_TRUE(hazards.needsBarrier({reads(&a, 1000, 100)})) << "overlaps the first slice";
    EXPECT_FALSE(hazards.needsBarrier({reads(&a, 0, 2048)}));
    EXPECT_TRUE(hazards.needsBarrier({writes(&a)})) << "the whole buffer";
}

TEST(HazardTracker, ExplicitBarrierClears) {
    auto hazards = HazardTracker{};
    EXPECT_FALSE(hazards.needsBarrier({writes(&a)}));
    hazards.barrierRecorded();
    EXPECT_FALSE(hazards.needsBarrier({reads(&a)}));
}

} // namespace
