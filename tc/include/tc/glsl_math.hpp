#pragma once

#include <cmath>
#include <cstring>
#include <type_traits>

#include <tc/vec.hpp>

// GLSL built-in functions. Each one accepts a scalar or a vector and works
// component-wise, exactly like its GLSL namesake, so the transpiler can emit
// the call unchanged.
namespace tc {

namespace detail {

template <class V>
inline constexpr bool isGenType = isScalar<ComponentOf<V>>;

template <class V>
inline constexpr bool isGenFloat = isGenType<V> and std::is_floating_point_v<ComponentOf<V>>;

template <class V>
inline constexpr bool isGenNumeric = isGenType<V> and not std::is_same_v<ComponentOf<V>, bool>;

template <class V>
inline constexpr bool isGenBool = isGenType<V> and std::is_same_v<ComponentOf<V>, bool>;

template <class T>
struct Identity {
    using type = T;
};

// Blocks template deduction so `min(v, 0.0f)` deduces only from `v`.
template <class T>
using NonDeduced = typename Identity<T>::type;

template <class V, class F>
constexpr auto componentwise(V const &v, F const f) noexcept {
    if constexpr (VecTraits<V>::isVector) {
        using R = decltype(f(v.x));
        auto result = typename VecTraits<V>::template Rebind<R>{};
        for (int i = 0; i < V::length(); ++i) {
            result.component(i) = f(v.component(i));
        }
        return result;
    } else {
        return f(v);
    }
}

template <class V, class F>
constexpr auto componentwise(V const &a, V const &b, F const f) noexcept {
    if constexpr (VecTraits<V>::isVector) {
        using R = decltype(f(a.x, b.x));
        auto result = typename VecTraits<V>::template Rebind<R>{};
        for (int i = 0; i < V::length(); ++i) {
            result.component(i) = f(a.component(i), b.component(i));
        }
        return result;
    } else {
        return f(a, b);
    }
}

template <class V, class F>
constexpr auto componentwise(V const &a, V const &b, V const &c, F const f) noexcept {
    if constexpr (VecTraits<V>::isVector) {
        using R = decltype(f(a.x, b.x, c.x));
        auto result = typename VecTraits<V>::template Rebind<R>{};
        for (int i = 0; i < V::length(); ++i) {
            result.component(i) = f(a.component(i), b.component(i), c.component(i));
        }
        return result;
    } else {
        return f(a, b, c);
    }
}

template <class V>
constexpr V splat(ComponentOf<V> const s) noexcept {
    if constexpr (VecTraits<V>::isVector) {
        return V{s};
    } else {
        return s;
    }
}

} // namespace detail

template <class V>
using EnableGenFloat = std::enable_if_t<detail::isGenFloat<V>, int>;
template <class V>
using EnableGenNumeric = std::enable_if_t<detail::isGenNumeric<V>, int>;

// Angle and trigonometry functions.
template <class V, EnableGenFloat<V> = 0>
V radians(V const &degrees) noexcept {
    using C = detail::ComponentOf<V>;
    return detail::componentwise(degrees, [](C const d) { return d * static_cast<C>(0.017453292519943295); });
}
template <class V, EnableGenFloat<V> = 0>
V degrees(V const &radians) noexcept {
    using C = detail::ComponentOf<V>;
    return detail::componentwise(radians, [](C const r) { return r * static_cast<C>(57.29577951308232); });
}
template <class V, EnableGenFloat<V> = 0>
V sin(V const &x) noexcept {
    return detail::componentwise(x, [](auto const c) { return std::sin(c); });
}
template <class V, EnableGenFloat<V> = 0>
V cos(V const &x) noexcept {
    return detail::componentwise(x, [](auto const c) { return std::cos(c); });
}
template <class V, EnableGenFloat<V> = 0>
V tan(V const &x) noexcept {
    return detail::componentwise(x, [](auto const c) { return std::tan(c); });
}
template <class V, EnableGenFloat<V> = 0>
V asin(V const &x) noexcept {
    return detail::componentwise(x, [](auto const c) { return std::asin(c); });
}
template <class V, EnableGenFloat<V> = 0>
V acos(V const &x) noexcept {
    return detail::componentwise(x, [](auto const c) { return std::acos(c); });
}
template <class V, EnableGenFloat<V> = 0>
V atan(V const &yOverX) noexcept {
    return detail::componentwise(yOverX, [](auto const c) { return std::atan(c); });
}
template <class V, EnableGenFloat<V> = 0>
V atan(V const &y, V const &x) noexcept {
    return detail::componentwise(y, x, [](auto const a, auto const b) { return std::atan2(a, b); });
}

// Exponential functions.
template <class V, EnableGenFloat<V> = 0>
V pow(V const &x, V const &y) noexcept {
    return detail::componentwise(x, y, [](auto const a, auto const b) { return std::pow(a, b); });
}
template <class V, EnableGenFloat<V> = 0>
V exp(V const &x) noexcept {
    return detail::componentwise(x, [](auto const c) { return std::exp(c); });
}
template <class V, EnableGenFloat<V> = 0>
V log(V const &x) noexcept {
    return detail::componentwise(x, [](auto const c) { return std::log(c); });
}
template <class V, EnableGenFloat<V> = 0>
V exp2(V const &x) noexcept {
    return detail::componentwise(x, [](auto const c) { return std::exp2(c); });
}
template <class V, EnableGenFloat<V> = 0>
V log2(V const &x) noexcept {
    return detail::componentwise(x, [](auto const c) { return std::log2(c); });
}
template <class V, EnableGenFloat<V> = 0>
V sqrt(V const &x) noexcept {
    return detail::componentwise(x, [](auto const c) { return std::sqrt(c); });
}
template <class V, EnableGenFloat<V> = 0>
V inversesqrt(V const &x) noexcept {
    using C = detail::ComponentOf<V>;
    return detail::componentwise(x, [](C const c) { return C{1} / std::sqrt(c); });
}

// Common functions.
template <class V, EnableGenNumeric<V> = 0>
V abs(V const &x) noexcept {
    using C = detail::ComponentOf<V>;
    return detail::componentwise(x, [](C const c) { return c < C{0} ? static_cast<C>(-c) : c; });
}
template <class V, EnableGenNumeric<V> = 0>
V sign(V const &x) noexcept {
    using C = detail::ComponentOf<V>;
    return detail::componentwise(x, [](C const c) {
        return c > C{0} ? C{1} : (c < C{0} ? static_cast<C>(-1) : C{0});
    });
}
template <class V, EnableGenFloat<V> = 0>
V floor(V const &x) noexcept {
    return detail::componentwise(x, [](auto const c) { return std::floor(c); });
}
template <class V, EnableGenFloat<V> = 0>
V ceil(V const &x) noexcept {
    return detail::componentwise(x, [](auto const c) { return std::ceil(c); });
}
template <class V, EnableGenFloat<V> = 0>
V trunc(V const &x) noexcept {
    return detail::componentwise(x, [](auto const c) { return std::trunc(c); });
}
template <class V, EnableGenFloat<V> = 0>
V round(V const &x) noexcept {
    return detail::componentwise(x, [](auto const c) { return std::round(c); });
}
template <class V, EnableGenFloat<V> = 0>
V fract(V const &x) noexcept {
    return detail::componentwise(x, [](auto const c) { return c - std::floor(c); });
}
template <class V, EnableGenFloat<V> = 0>
V mod(V const &x, V const &y) noexcept {
    return detail::componentwise(x, y, [](auto const a, auto const b) { return a - b * std::floor(a / b); });
}
template <class T, int N, EnableGenFloat<T> = 0>
vec<T, N> mod(vec<T, N> const &x, detail::NonDeduced<T> const y) noexcept {
    return mod(x, vec<T, N>{y});
}
template <class V, EnableGenNumeric<V> = 0>
constexpr V min(V const &x, V const &y) noexcept {
    return detail::componentwise(x, y, [](auto const a, auto const b) { return b < a ? b : a; });
}
template <class T, int N, EnableGenNumeric<T> = 0>
constexpr vec<T, N> min(vec<T, N> const &x, detail::NonDeduced<T> const y) noexcept {
    return min(x, vec<T, N>{y});
}
template <class V, EnableGenNumeric<V> = 0>
constexpr V max(V const &x, V const &y) noexcept {
    return detail::componentwise(x, y, [](auto const a, auto const b) { return a < b ? b : a; });
}
template <class T, int N, EnableGenNumeric<T> = 0>
constexpr vec<T, N> max(vec<T, N> const &x, detail::NonDeduced<T> const y) noexcept {
    return max(x, vec<T, N>{y});
}
template <class V, EnableGenNumeric<V> = 0>
constexpr V clamp(V const &x, V const &lo, V const &hi) noexcept {
    return min(max(x, lo), hi);
}
template <class T, int N, EnableGenNumeric<T> = 0>
constexpr vec<T, N> clamp(vec<T, N> const &x, detail::NonDeduced<T> const lo,
                          detail::NonDeduced<T> const hi) noexcept {
    return clamp(x, vec<T, N>{lo}, vec<T, N>{hi});
}
template <class V, EnableGenFloat<V> = 0>
constexpr V mix(V const &x, V const &y, V const &a) noexcept {
    return detail::componentwise(x, y, a, [](auto const l, auto const r, auto const t) { return l * (1 - t) + r * t; });
}
template <class T, int N, EnableGenFloat<T> = 0>
constexpr vec<T, N> mix(vec<T, N> const &x, vec<T, N> const &y, detail::NonDeduced<T> const a) noexcept {
    return mix(x, y, vec<T, N>{a});
}
template <class V, EnableGenFloat<V> = 0>
constexpr V step(V const &edge, V const &x) noexcept {
    using C = detail::ComponentOf<V>;
    return detail::componentwise(edge, x, [](C const e, C const c) { return c < e ? C{0} : C{1}; });
}
template <class T, int N, EnableGenFloat<T> = 0>
constexpr vec<T, N> step(detail::NonDeduced<T> const edge, vec<T, N> const &x) noexcept {
    return step(vec<T, N>{edge}, x);
}
template <class V, EnableGenFloat<V> = 0>
constexpr V smoothstep(V const &edge0, V const &edge1, V const &x) noexcept {
    using C = detail::ComponentOf<V>;
    return detail::componentwise(edge0, edge1, x, [](C const e0, C const e1, C const c) {
        auto const t = min((c - e0) / (e1 - e0), C{1});
        auto const clamped = max(t, C{0});
        return clamped * clamped * (C{3} - C{2} * clamped);
    });
}
template <class T, int N, EnableGenFloat<T> = 0>
constexpr vec<T, N> smoothstep(detail::NonDeduced<T> const edge0, detail::NonDeduced<T> const edge1,
                               vec<T, N> const &x) noexcept {
    return smoothstep(vec<T, N>{edge0}, vec<T, N>{edge1}, x);
}
template <class V, EnableGenFloat<V> = 0>
auto isnan(V const &x) noexcept {
    return detail::componentwise(x, [](auto const c) { return std::isnan(c); });
}
template <class V, EnableGenFloat<V> = 0>
auto isinf(V const &x) noexcept {
    return detail::componentwise(x, [](auto const c) { return std::isinf(c); });
}

// Bit reinterpretation, handy for hashing in kernels.
template <class V, std::enable_if_t<std::is_same_v<detail::ComponentOf<V>, float>, int> = 0>
auto floatBitsToUint(V const &x) noexcept {
    return detail::componentwise(x, [](float const c) {
        auto bits = std::uint32_t{};
        std::memcpy(&bits, &c, sizeof bits);
        return bits;
    });
}
template <class V, std::enable_if_t<std::is_same_v<detail::ComponentOf<V>, float>, int> = 0>
auto floatBitsToInt(V const &x) noexcept {
    return detail::componentwise(x, [](float const c) {
        auto bits = std::int32_t{};
        std::memcpy(&bits, &c, sizeof bits);
        return bits;
    });
}
template <class V, std::enable_if_t<std::is_same_v<detail::ComponentOf<V>, std::uint32_t>, int> = 0>
auto uintBitsToFloat(V const &x) noexcept {
    return detail::componentwise(x, [](std::uint32_t const c) {
        auto value = 0.0f;
        std::memcpy(&value, &c, sizeof value);
        return value;
    });
}
template <class V, std::enable_if_t<std::is_same_v<detail::ComponentOf<V>, std::int32_t>, int> = 0>
auto intBitsToFloat(V const &x) noexcept {
    return detail::componentwise(x, [](std::int32_t const c) {
        auto value = 0.0f;
        std::memcpy(&value, &c, sizeof value);
        return value;
    });
}

// Geometric functions.
template <class V, EnableGenFloat<V> = 0>
constexpr detail::ComponentOf<V> dot(V const &x, V const &y) noexcept {
    if constexpr (detail::VecTraits<V>::isVector) {
        auto sum = detail::ComponentOf<V>{0};
        for (int i = 0; i < V::length(); ++i) {
            sum += x.component(i) * y.component(i);
        }
        return sum;
    } else {
        return x * y;
    }
}
template <class V, EnableGenFloat<V> = 0>
detail::ComponentOf<V> length(V const &x) noexcept {
    return std::sqrt(dot(x, x));
}
template <class V, EnableGenFloat<V> = 0>
detail::ComponentOf<V> distance(V const &p0, V const &p1) noexcept {
    return length(p0 - p1);
}
template <class V, EnableGenFloat<V> = 0>
V normalize(V const &x) noexcept {
    return x / detail::splat<V>(length(x));
}
template <class T, EnableGenFloat<T> = 0>
constexpr vec<T, 3> cross(vec<T, 3> const &a, vec<T, 3> const &b) noexcept {
    return vec<T, 3>{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
template <class V, EnableGenFloat<V> = 0>
V faceforward(V const &n, V const &i, V const &nRef) noexcept {
    return dot(nRef, i) < 0 ? n : -n;
}
template <class V, EnableGenFloat<V> = 0>
V reflect(V const &i, V const &n) noexcept {
    return i - detail::splat<V>(2 * dot(n, i)) * n;
}
template <class V, EnableGenFloat<V> = 0>
V refract(V const &i, V const &n, detail::ComponentOf<V> const eta) noexcept {
    using C = detail::ComponentOf<V>;
    auto const cosI = dot(n, i);
    auto const k = C{1} - eta * eta * (C{1} - cosI * cosI);
    if (k < C{0}) {
        return V{};
    }
    return detail::splat<V>(eta) * i - detail::splat<V>(eta * cosI + std::sqrt(k)) * n;
}

// Vector relational functions.
template <class T, int N>
constexpr vec<bool, N> lessThan(vec<T, N> const &x, vec<T, N> const &y) noexcept {
    return detail::componentwise(x, y, [](T const a, T const b) { return a < b; });
}
template <class T, int N>
constexpr vec<bool, N> lessThanEqual(vec<T, N> const &x, vec<T, N> const &y) noexcept {
    return detail::componentwise(x, y, [](T const a, T const b) { return a <= b; });
}
template <class T, int N>
constexpr vec<bool, N> greaterThan(vec<T, N> const &x, vec<T, N> const &y) noexcept {
    return detail::componentwise(x, y, [](T const a, T const b) { return a > b; });
}
template <class T, int N>
constexpr vec<bool, N> greaterThanEqual(vec<T, N> const &x, vec<T, N> const &y) noexcept {
    return detail::componentwise(x, y, [](T const a, T const b) { return a >= b; });
}
template <class T, int N>
constexpr vec<bool, N> equal(vec<T, N> const &x, vec<T, N> const &y) noexcept {
    return detail::componentwise(x, y, [](T const a, T const b) { return a == b; });
}
template <class T, int N>
constexpr vec<bool, N> notEqual(vec<T, N> const &x, vec<T, N> const &y) noexcept {
    return detail::componentwise(x, y, [](T const a, T const b) { return a != b; });
}
template <int N>
constexpr bool any(vec<bool, N> const &x) noexcept {
    for (int i = 0; i < N; ++i) {
        if (x.component(i)) {
            return true;
        }
    }
    return false;
}
template <int N>
constexpr bool all(vec<bool, N> const &x) noexcept {
    for (int i = 0; i < N; ++i) {
        if (not x.component(i)) {
            return false;
        }
    }
    return true;
}
// GLSL `not(bvec)`; `not` is a reserved alternative token in C++.
template <int N>
constexpr vec<bool, N> not_(vec<bool, N> const &x) noexcept {
    return detail::componentwise(x, [](bool const c) { return not c; });
}

} // namespace tc
