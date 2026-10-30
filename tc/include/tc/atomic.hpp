#pragma once

#include <type_traits>

#include <tc/vec.hpp>

#if not(defined(__GNUC__) or defined(__clang__))
#error "tc atomics use the GCC/Clang __atomic built-ins"
#endif

// GLSL atomic memory functions. `memory` must be a buffer element or a
// Shared variable (the transpiler checks this). They return the previous
// value. On the CPU, workgroups run on several threads, so these are real
// atomic operations there too; like GLSL they impose no ordering on other
// memory accesses.
namespace tc {

namespace detail {

template <class T>
using EnableAtomic =
    std::enable_if_t<isIntegralScalar<T> or std::is_same_v<T, i64> or std::is_same_v<T, u64>, int>;

template <class T>
struct AtomicIdentity {
    using type = T;
};

template <class T, class F>
T atomicUpdate(T &memory, F const update) noexcept {
    auto expected = __atomic_load_n(&memory, __ATOMIC_RELAXED);
    while (not __atomic_compare_exchange_n(&memory, &expected, update(expected), true, __ATOMIC_RELAXED,
                                           __ATOMIC_RELAXED)) {
    }
    return expected;
}

} // namespace detail

template <class T, detail::EnableAtomic<T> = 0>
T atomicAdd(T &memory, typename detail::AtomicIdentity<T>::type const data) noexcept {
    return __atomic_fetch_add(&memory, data, __ATOMIC_RELAXED);
}

template <class T, detail::EnableAtomic<T> = 0>
T atomicAnd(T &memory, typename detail::AtomicIdentity<T>::type const data) noexcept {
    return __atomic_fetch_and(&memory, data, __ATOMIC_RELAXED);
}

template <class T, detail::EnableAtomic<T> = 0>
T atomicOr(T &memory, typename detail::AtomicIdentity<T>::type const data) noexcept {
    return __atomic_fetch_or(&memory, data, __ATOMIC_RELAXED);
}

template <class T, detail::EnableAtomic<T> = 0>
T atomicXor(T &memory, typename detail::AtomicIdentity<T>::type const data) noexcept {
    return __atomic_fetch_xor(&memory, data, __ATOMIC_RELAXED);
}

template <class T, detail::EnableAtomic<T> = 0>
T atomicMin(T &memory, typename detail::AtomicIdentity<T>::type const data) noexcept {
    return detail::atomicUpdate(memory, [data](T const current) { return data < current ? data : current; });
}

template <class T, detail::EnableAtomic<T> = 0>
T atomicMax(T &memory, typename detail::AtomicIdentity<T>::type const data) noexcept {
    return detail::atomicUpdate(memory, [data](T const current) { return current < data ? data : current; });
}

template <class T, detail::EnableAtomic<T> = 0>
T atomicExchange(T &memory, typename detail::AtomicIdentity<T>::type const data) noexcept {
    return __atomic_exchange_n(&memory, data, __ATOMIC_RELAXED);
}

// Stores `data` only if `memory` equals `compare`; returns the old value.
template <class T, detail::EnableAtomic<T> = 0>
T atomicCompSwap(T &memory, typename detail::AtomicIdentity<T>::type const compare,
                 typename detail::AtomicIdentity<T>::type const data) noexcept {
    auto expected = compare;
    __atomic_compare_exchange_n(&memory, &expected, data, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
    return expected;
}

} // namespace tc
