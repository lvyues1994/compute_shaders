#pragma once

#include <array>
#include <cstdint>

#include <tc/tc.hpp>

// Kernels for subgroup operations and 64-bit integers. Their results do not
// depend on the subgroup size, so every device must agree with the CPU.
namespace tests {

using namespace tc;

struct TC_KERNEL SubgroupOps {
    static constexpr char fileLocation[] = "subgroup_ops";
    static constexpr bool usesSubgroups = true;
    static constexpr bool usesBarrier = true;
    uvec3 local_size{64, 1, 1};

    BufferBinding<uint, 0> values;
    BufferBinding<uint, 1> prefix;    // inclusive prefix sum within the workgroup
    BufferBinding<uint, 2> sums;      // per workgroup
    BufferBinding<uint, 3> compacted; // the odd values, in any order
    BufferBinding<uint, 4> counters;  // [0] compacted, [1] maximum, [2] subgroups
    BufferBinding<uint, 5> checks;    // per invocation: one bit per lane-relative check that held
    Shared<std::array<uint, 64>> subgroupTotals;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        uint lane = gl_SubgroupInvocationID;
        uint v = values[i];

        // Workgroup prefix sum: scans within subgroups, then their totals.
        uint inclusive = subgroupInclusiveAdd(v);
        uint last = subgroupBallotFindMSB(subgroupBallot(true));
        if (lane == last) {
            subgroupTotals[gl_SubgroupID] = inclusive;
        }
        barrier();
        uint offset = 0u;
        for (uint s = 0u; s < gl_SubgroupID; ++s) {
            offset += subgroupTotals[s];
        }
        prefix[i] = offset + inclusive;

        // Reductions, one atomic per subgroup.
        uint total = subgroupAdd(v);
        uint largest = subgroupMax(v);
        if (subgroupElect()) {
            atomicAdd(sums[gl_WorkGroupID.x], total);
            atomicMax(counters[1], largest);
            atomicAdd(counters[2], 1u);
        }

        // Stream compaction with a ballot.
        bool keep = (v & 1u) != 0u;
        uvec4 kept = subgroupBallot(keep);
        uint base = 0u;
        if (subgroupElect()) {
            base = atomicAdd(counters[0], subgroupBallotBitCount(kept));
        }
        base = subgroupBroadcastFirst(base);
        if (keep) {
            compacted[base + subgroupBallotExclusiveBitCount(kept)] = v;
        }

        // Checks relative to this invocation's lane.
        uint first = i - lane;
        uint passed = 0u;
        passed |= subgroupAll(true) ? 1u : 0u;
        passed |= subgroupAny(lane == 0u) ? 2u : 0u;
        passed |= subgroupAllEqual(gl_SubgroupID) ? 4u : 0u;
        passed |= subgroupBroadcastFirst(i) == first ? 8u : 0u;
        passed |= subgroupBroadcast(i, 0u) == first ? 16u : 0u;
        passed |= gl_SubgroupSize < 2u or subgroupShuffleXor(i, 1u) == (i ^ 1u) ? 32u : 0u;
        uint down = subgroupShuffleDown(i, 1u);
        passed |= lane == last or down == i + 1u ? 64u : 0u;
        uint up = subgroupShuffleUp(i, 1u);
        passed |= lane == 0u or up == i - 1u ? 128u : 0u;
        passed |= subgroupInclusiveAdd(1.0f) == float(lane + 1u) ? 256u : 0u;
        passed |= subgroupExclusiveMin(int(lane)) == (lane == 0u ? 2147483647 : 0) ? 512u : 0u;
        vec2 spread = subgroupMax(vec2(float(lane), -float(lane)));
        passed |= spread.x == float(last) and spread.y == 0.0f ? 1024u : 0u;
        passed |= subgroupInverseBallot(gl_SubgroupEqMask) ? 2048u : 0u;
        passed |= subgroupBallotBitCount(gl_SubgroupLtMask & kept) == subgroupBallotExclusiveBitCount(kept) ? 4096u : 0u;
        passed |= subgroupShuffle(i, lane) == i ? 8192u : 0u;
        checks[i] = passed;
    }
};

// 64-bit integer arithmetic.
struct TC_KERNEL Wide64 {
    static constexpr char fileLocation[] = "wide_64";
    uvec3 local_size{64, 1, 1};

    BufferBinding<u64, 0> keys;
    BufferBinding<i64, 1> out;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        u64 k = keys[i];
        u64 mixed = (k * 6364136223846793005ull + 1442695040888963407ull) >> 7u;
        i64 centred = i64(mixed % 1000000007ull) - 500000000ll;
        out[i * 2u] = centred * -3ll;
        out[i * 2u + 1u] = i64((u64(i) << 40u) | (mixed & 0xffffffffffull));
    }
};

// The visibility buffer's single pass: depth in the high bits and the
// triangle in the low bits, kept by a 64-bit atomicMax per pixel.
struct TC_KERNEL VisibilityBuffer {
    static constexpr char fileLocation[] = "visibility_buffer";
    uvec3 local_size{64, 1, 1};

    BufferBinding<u64, 0> pixels;

    void main() {
        uint triangle = gl_GlobalInvocationID.x;
        uint pixel = (triangle * 7u) % uint(pixels.length());
        uint depth = (triangle * 2654435761u) >> 8u;
        atomicMax(pixels[pixel], (u64(depth) << 32u) | u64(triangle));
    }
};

} // namespace tests
