#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include <tc/vec.hpp>

// GLSL subgroup built-ins (GL_KHR_shader_subgroup: basic, vote, arithmetic,
// ballot, shuffle and relative shuffle). Kernels that use them declare
// `static constexpr bool usesSubgroups = true;`; GPU devices run them where
// DeviceLimits::features has KernelFeature::Subgroups.
//
// The CPU backend runs a workgroup's invocations as fibers in subgroups of
// CpuDeviceConfig::subgroupSize. Each operation waits until every active
// invocation of the subgroup has reached it; invocations that returned
// early are inactive, as on GPUs. Operations must be reached by all active
// invocations (subgroup-uniform control flow): one inside a branch that
// only some invocations take is not supported, and the CPU backend aborts
// when invocations of a subgroup wait at different operations.
//
// Results that depend on the subgroup size (ballot bits, shuffles) differ
// between devices; gl_SubgroupSize tells kernels which size they run with.
namespace tc {

inline thread_local uint gl_SubgroupSize{};
inline thread_local uint gl_SubgroupInvocationID{};
inline thread_local uint gl_SubgroupID{};
inline thread_local uint gl_NumSubgroups{};
inline thread_local uvec4 gl_SubgroupEqMask{};
inline thread_local uvec4 gl_SubgroupGeMask{};
inline thread_local uvec4 gl_SubgroupGtMask{};
inline thread_local uvec4 gl_SubgroupLeMask{};
inline thread_local uvec4 gl_SubgroupLtMask{};

namespace detail {

inline constexpr std::uint32_t kMaxSubgroupSize = 128;

// The active invocations of one subgroup waiting at one operation, in lane
// order. Inputs and outputs live on their fiber stacks.
struct SubgroupLanes {
    std::uint32_t count = 0;
    std::uint32_t lane[kMaxSubgroupSize] = {};
    void const *input[kMaxSubgroupSize] = {};
    void *output[kMaxSubgroupSize] = {};
};

using SubgroupResolve = void (*)(SubgroupLanes const &lanes);

// Suspends the invocation until every active invocation of its subgroup has
// called it with the same `resolve`, which then writes all their outputs.
void subgroupRendezvous(void const *input, void *output, SubgroupResolve resolve);

// Sets the subgroup built-ins of workgroup invocation `index`.
void enterSubgroup(std::uint32_t index, std::uint32_t invocations) noexcept;

// The CPU backend's subgroup size on this thread.
std::uint32_t &subgroupSizeOnThisThread() noexcept;

template <class T>
struct SubgroupScalar {
    using type = T;
};
template <class T, int N>
struct SubgroupScalar<vec<T, N>> {
    using type = T;
};

template <class T>
inline constexpr int subgroupComponents = 1;
template <class T, int N>
inline constexpr int subgroupComponents<vec<T, N>> = N;

template <class T>
inline constexpr bool isSubgroupValue = std::is_same_v<typename SubgroupScalar<T>::type, float>
                                     or std::is_same_v<typename SubgroupScalar<T>::type, std::int32_t>
                                     or std::is_same_v<typename SubgroupScalar<T>::type, std::uint32_t>
                                     or std::is_same_v<typename SubgroupScalar<T>::type, bool>
                                     or std::is_same_v<typename SubgroupScalar<T>::type, long long>
                                     or std::is_same_v<typename SubgroupScalar<T>::type, unsigned long long>;

template <class T>
using EnableSubgroup = std::enable_if_t<isSubgroupValue<T>, int>;

// Component-wise operations with their identities.
template <class Scalar>
struct Add {
    static Scalar apply(Scalar const a, Scalar const b) noexcept { return a + b; }
    static Scalar identity() noexcept { return Scalar(0); }
};
template <class Scalar>
struct Mul {
    static Scalar apply(Scalar const a, Scalar const b) noexcept { return a * b; }
    static Scalar identity() noexcept { return Scalar(1); }
};
template <class Scalar>
struct Min {
    static Scalar apply(Scalar const a, Scalar const b) noexcept { return b < a ? b : a; }
    static Scalar identity() noexcept {
        return std::numeric_limits<Scalar>::has_infinity ? std::numeric_limits<Scalar>::infinity()
                                                         : std::numeric_limits<Scalar>::max();
    }
};
template <class Scalar>
struct Max {
    static Scalar apply(Scalar const a, Scalar const b) noexcept { return a < b ? b : a; }
    static Scalar identity() noexcept {
        return std::numeric_limits<Scalar>::has_infinity ? -std::numeric_limits<Scalar>::infinity()
                                                         : std::numeric_limits<Scalar>::lowest();
    }
};
template <class Scalar>
struct And {
    static Scalar apply(Scalar const a, Scalar const b) noexcept { return static_cast<Scalar>(a & b); }
    static Scalar identity() noexcept { return static_cast<Scalar>(~Scalar(0)); }
};
template <>
struct And<bool> {
    static bool apply(bool const a, bool const b) noexcept { return a and b; }
    static bool identity() noexcept { return true; }
};
template <class Scalar>
struct Or {
    static Scalar apply(Scalar const a, Scalar const b) noexcept { return static_cast<Scalar>(a | b); }
    static Scalar identity() noexcept { return Scalar(0); }
};
template <class Scalar>
struct Xor {
    static Scalar apply(Scalar const a, Scalar const b) noexcept { return static_cast<Scalar>(a ^ b); }
    static Scalar identity() noexcept { return Scalar(0); }
};

template <template <class> class Op, class T>
T combine(T const &a, T const &b) noexcept {
    using Scalar = typename SubgroupScalar<T>::type;
    if constexpr (subgroupComponents<T> == 1) {
        return Op<Scalar>::apply(a, b);
    } else {
        auto result = T{};
        for (int c = 0; c < subgroupComponents<T>; ++c) {
            result[c] = Op<Scalar>::apply(a[c], b[c]);
        }
        return result;
    }
}

template <template <class> class Op, class T>
T identity() noexcept {
    return T(Op<typename SubgroupScalar<T>::type>::identity());
}

enum class Scan : std::uint8_t {
    Reduce,
    Inclusive,
    Exclusive,
};

template <template <class> class Op, class T, Scan S>
void resolveScan(SubgroupLanes const &lanes) {
    auto running = identity<Op, T>();
    for (std::uint32_t i = 0; i < lanes.count; ++i) {
        auto const before = running;
        running = combine<Op>(running, *static_cast<T const *>(lanes.input[i]));
        if constexpr (S == Scan::Exclusive) {
            *static_cast<T *>(lanes.output[i]) = before;
        } else if constexpr (S == Scan::Inclusive) {
            *static_cast<T *>(lanes.output[i]) = running;
        }
    }
    if constexpr (S == Scan::Reduce) {
        for (std::uint32_t i = 0; i < lanes.count; ++i) {
            *static_cast<T *>(lanes.output[i]) = running;
        }
    }
}

template <template <class> class Op, Scan S, class T>
T scan(T const &value) {
    auto result = T{};
    subgroupRendezvous(&value, &result, &resolveScan<Op, T, S>);
    return result;
}

// A value with the lane it comes from (shuffles, broadcasts).
template <class T>
struct FromLane {
    T value;
    std::uint32_t source;
};

template <class T>
void resolveFromLane(SubgroupLanes const &lanes) {
    for (std::uint32_t i = 0; i < lanes.count; ++i) {
        auto const source = static_cast<FromLane<T> const *>(lanes.input[i])->source;
        auto result = T{};
        for (std::uint32_t j = 0; j < lanes.count; ++j) {
            if (lanes.lane[j] == source) {
                result = static_cast<FromLane<T> const *>(lanes.input[j])->value;
            }
        }
        *static_cast<T *>(lanes.output[i]) = result;
    }
}

template <class T>
T fromLane(T const &value, std::uint32_t const source) {
    auto const input = FromLane<T>{value, source};
    auto result = T{};
    subgroupRendezvous(&input, &result, &resolveFromLane<T>);
    return result;
}

inline void resolveBallot(SubgroupLanes const &lanes) {
    auto bits = uvec4{};
    for (std::uint32_t i = 0; i < lanes.count; ++i) {
        if (*static_cast<bool const *>(lanes.input[i])) {
            bits[static_cast<int>(lanes.lane[i] / 32)] |= 1u << (lanes.lane[i] % 32);
        }
    }
    for (std::uint32_t i = 0; i < lanes.count; ++i) {
        *static_cast<uvec4 *>(lanes.output[i]) = bits;
    }
}

template <class T>
void resolveAllEqual(SubgroupLanes const &lanes) {
    auto equal = true;
    for (std::uint32_t i = 1; i < lanes.count; ++i) {
        equal = equal and *static_cast<T const *>(lanes.input[i]) == *static_cast<T const *>(lanes.input[0]);
    }
    for (std::uint32_t i = 0; i < lanes.count; ++i) {
        *static_cast<bool *>(lanes.output[i]) = equal;
    }
}

inline void resolveElect(SubgroupLanes const &lanes) {
    for (std::uint32_t i = 0; i < lanes.count; ++i) {
        *static_cast<bool *>(lanes.output[i]) = i == 0;
    }
}

inline void resolveBarrier(SubgroupLanes const &) {}

inline uint popcount(uint const v) noexcept {
    return static_cast<uint>(__builtin_popcount(v));
}

// Bits of `value` for lanes below `end` (at most gl_SubgroupSize).
inline uint countBelow(uvec4 const &value, uint const end) noexcept {
    auto count = 0u;
    for (uint word = 0; word < 4; ++word) {
        auto const first = word * 32u;
        if (end <= first) {
            break;
        }
        auto const bits = end - first >= 32u ? 32u : end - first;
        auto const mask = bits == 32u ? ~0u : (1u << bits) - 1u;
        count += popcount(value[static_cast<int>(word)] & mask);
    }
    return count;
}

} // namespace detail

// Basic.
inline void subgroupBarrier() {
    auto none = 0;
    detail::subgroupRendezvous(&none, &none, &detail::resolveBarrier);
}
inline void subgroupMemoryBarrier() noexcept {}
inline bool subgroupElect() {
    auto none = 0;
    auto elected = false;
    detail::subgroupRendezvous(&none, &elected, &detail::resolveElect);
    return elected;
}

// Vote.
inline bool subgroupAll(bool const value) {
    return detail::scan<detail::And, detail::Scan::Reduce>(value);
}
inline bool subgroupAny(bool const value) {
    return detail::scan<detail::Or, detail::Scan::Reduce>(value);
}
template <class T, detail::EnableSubgroup<T> = 0>
bool subgroupAllEqual(T const &value) {
    auto equal = false;
    detail::subgroupRendezvous(&value, &equal, &detail::resolveAllEqual<T>);
    return equal;
}

// Arithmetic: reductions and inclusive / exclusive scans in lane order.
#define TC_SUBGROUP_ARITHMETIC(Name, Op)                                                                               \
    template <class T, detail::EnableSubgroup<T> = 0>                                                                  \
    T subgroup##Name(T const &value) {                                                                                 \
        return detail::scan<detail::Op, detail::Scan::Reduce>(value);                                                  \
    }                                                                                                                  \
    template <class T, detail::EnableSubgroup<T> = 0>                                                                  \
    T subgroupInclusive##Name(T const &value) {                                                                        \
        return detail::scan<detail::Op, detail::Scan::Inclusive>(value);                                               \
    }                                                                                                                  \
    template <class T, detail::EnableSubgroup<T> = 0>                                                                  \
    T subgroupExclusive##Name(T const &value) {                                                                        \
        return detail::scan<detail::Op, detail::Scan::Exclusive>(value);                                               \
    }
TC_SUBGROUP_ARITHMETIC(Add, Add)
TC_SUBGROUP_ARITHMETIC(Mul, Mul)
TC_SUBGROUP_ARITHMETIC(Min, Min)
TC_SUBGROUP_ARITHMETIC(Max, Max)
TC_SUBGROUP_ARITHMETIC(And, And)
TC_SUBGROUP_ARITHMETIC(Or, Or)
TC_SUBGROUP_ARITHMETIC(Xor, Xor)
#undef TC_SUBGROUP_ARITHMETIC

// Ballot.
inline uvec4 subgroupBallot(bool const value) {
    auto bits = uvec4{};
    detail::subgroupRendezvous(&value, &bits, &detail::resolveBallot);
    return bits;
}
inline bool subgroupInverseBallot(uvec4 const &value) noexcept {
    return (value[static_cast<int>(gl_SubgroupInvocationID / 32u)] >> (gl_SubgroupInvocationID % 32u) & 1u) != 0u;
}
inline bool subgroupBallotBitExtract(uvec4 const &value, uint const index) noexcept {
    return (value[static_cast<int>(index / 32u)] >> (index % 32u) & 1u) != 0u;
}
inline uint subgroupBallotBitCount(uvec4 const &value) noexcept {
    return detail::countBelow(value, gl_SubgroupSize);
}
inline uint subgroupBallotInclusiveBitCount(uvec4 const &value) noexcept {
    return detail::countBelow(value, gl_SubgroupInvocationID + 1u);
}
inline uint subgroupBallotExclusiveBitCount(uvec4 const &value) noexcept {
    return detail::countBelow(value, gl_SubgroupInvocationID);
}
inline uint subgroupBallotFindLSB(uvec4 const &value) noexcept {
    for (uint i = 0; i < gl_SubgroupSize; ++i) {
        if (subgroupBallotBitExtract(value, i)) {
            return i;
        }
    }
    return 0u;
}
inline uint subgroupBallotFindMSB(uvec4 const &value) noexcept {
    for (uint i = gl_SubgroupSize; i-- > 0;) {
        if (subgroupBallotBitExtract(value, i)) {
            return i;
        }
    }
    return 0u;
}
// `id` must be the same in every invocation (a constant in GLSL).
template <class T, detail::EnableSubgroup<T> = 0>
T subgroupBroadcast(T const &value, uint const id) {
    return detail::fromLane(value, id);
}
template <class T, detail::EnableSubgroup<T> = 0>
T subgroupBroadcastFirst(T const &value) {
    auto const first = subgroupBallotFindLSB(subgroupBallot(true));
    return detail::fromLane(value, first);
}

// Shuffle: values of other lanes; reading an inactive lane gives zero here
// and an undefined value on GPUs.
template <class T, detail::EnableSubgroup<T> = 0>
T subgroupShuffle(T const &value, uint const id) {
    return detail::fromLane(value, id);
}
template <class T, detail::EnableSubgroup<T> = 0>
T subgroupShuffleXor(T const &value, uint const mask) {
    return detail::fromLane(value, gl_SubgroupInvocationID ^ mask);
}
template <class T, detail::EnableSubgroup<T> = 0>
T subgroupShuffleUp(T const &value, uint const delta) {
    return detail::fromLane(value, gl_SubgroupInvocationID - delta);
}
template <class T, detail::EnableSubgroup<T> = 0>
T subgroupShuffleDown(T const &value, uint const delta) {
    return detail::fromLane(value, gl_SubgroupInvocationID + delta);
}

// `static constexpr bool usesSubgroups = true;` marks kernels whose
// invocations exchange values through subgroup operations.
template <class K, class = void>
struct UsesSubgroups : std::false_type {};

template <class K>
struct UsesSubgroups<K, std::void_t<decltype(K::usesSubgroups)>> : std::bool_constant<K::usesSubgroups> {};

} // namespace tc
