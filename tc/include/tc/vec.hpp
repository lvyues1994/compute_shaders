#pragma once

#include <cassert>
#include <cstdint>
#include <type_traits>

#include <tc/detail/vec_fields.hpp>

namespace tc {

using uint = std::uint32_t;

// The GLSL scalar types a kernel may use; `int`/`uint` are exactly 32 bits.
template <class T>
inline constexpr bool isScalar = std::is_same_v<T, bool> or std::is_same_v<T, std::int32_t>
    or std::is_same_v<T, std::uint32_t> or std::is_same_v<T, float>
    or std::is_same_v<T, double>;

template <class T>
inline constexpr bool isIntegralScalar = std::is_same_v<T, std::int32_t>
    or std::is_same_v<T, std::uint32_t>;

namespace detail {

// How many vector components an argument of a GLSL-style constructor such as
// `vec4(xy, z, 1.0f)` contributes. Specialised for vectors here and for
// Uniform in uniform.hpp.
template <class A, class = void>
struct ComponentSource {
    static constexpr int count = -100;
};

template <class A>
struct ComponentSource<A, std::enable_if_t<isScalar<A>>> {
    static constexpr int count = 1;

    template <class V>
    static constexpr void append(V &target, int &cursor, A const value) noexcept {
        target.component(cursor++) = static_cast<typename V::value_type>(value);
    }
};

template <class U, int M>
struct ComponentSource<vec<U, M>> {
    static constexpr int count = M;

    template <class V>
    static constexpr void append(V &target, int &cursor, vec<U, M> const &value) noexcept {
        for (int i = 0; i < M; ++i) {
            target.component(cursor++) = static_cast<typename V::value_type>(value.component(i));
        }
    }
};

template <class A>
using ComponentSourceOf = ComponentSource<std::remove_cv_t<std::remove_reference_t<A>>>;

template <class V>
struct VecTraits {
    static constexpr bool isVector = false;
    using Component = V;
    template <class U>
    using Rebind = U;
};

template <class T, int N>
struct VecTraits<vec<T, N>> {
    static constexpr bool isVector = true;
    using Component = T;
    template <class U>
    using Rebind = vec<U, N>;
};

template <class V>
using ComponentOf = typename VecTraits<V>::Component;

} // namespace detail

template <class T, int N>
struct vec : detail::VecFields<T, N> {
    static_assert(isScalar<T>, "vector components must be bool, int32_t, uint32_t, float or double");
    static_assert(N >= 2 and N <= 4, "GLSL vectors have 2, 3 or 4 components");

    using value_type = T;

    constexpr vec() noexcept = default;

    // vec3(1.0f): broadcast one scalar to every component.
    template <class S, std::enable_if_t<detail::ComponentSourceOf<S>::count == 1, int> = 0>
    constexpr explicit vec(S const &scalar) noexcept {
        auto cursor = 0;
        detail::ComponentSourceOf<S>::append(*this, cursor, scalar);
        for (int i = 1; i < N; ++i) {
            this->component(i) = this->component(0);
        }
    }

    // vec4(xy, z, 1.0f): concatenate components of scalars and vectors.
    template <class... Args,
              std::enable_if_t<(sizeof...(Args) >= 2)
                                   and ((detail::ComponentSourceOf<Args>::count + ...) == N),
                               int> = 0>
    constexpr vec(Args const &...args) noexcept {
        int cursor = 0;
        (detail::ComponentSourceOf<Args>::append(*this, cursor, args), ...);
    }

    // vec3(ivec3) converts element types; vec3(vec4) drops trailing components.
    template <class U, int M,
              std::enable_if_t<(M >= N) and not(M == N and std::is_same_v<U, T>), int> = 0>
    constexpr explicit vec(vec<U, M> const &other) noexcept {
        for (int i = 0; i < N; ++i) {
            this->component(i) = static_cast<T>(other.component(i));
        }
    }

    static constexpr int length() noexcept { return N; }

    template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
    constexpr T &operator[](I const index) noexcept {
        assert(index >= 0 and static_cast<int>(index) < N);
        return this->component(static_cast<int>(index));
    }

    template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
    constexpr T const &operator[](I const index) const noexcept {
        assert(index >= 0 and static_cast<int>(index) < N);
        return this->component(static_cast<int>(index));
    }

    friend constexpr vec operator+(vec const &a) noexcept { return a; }
    friend constexpr vec operator-(vec const &a) noexcept {
        return map(a, [](T const c) { return static_cast<T>(-c); });
    }
    friend constexpr vec operator~(vec const &a) noexcept {
        static_assert(isIntegralScalar<T>, "operator~ requires an integer vector");
        return map(a, [](T const c) { return static_cast<T>(~c); });
    }

    friend constexpr vec operator+(vec const &a, vec const &b) noexcept {
        return zip(a, b, [](T const l, T const r) { return l + r; });
    }
    friend constexpr vec operator-(vec const &a, vec const &b) noexcept {
        return zip(a, b, [](T const l, T const r) { return l - r; });
    }
    friend constexpr vec operator*(vec const &a, vec const &b) noexcept {
        return zip(a, b, [](T const l, T const r) { return l * r; });
    }
    friend constexpr vec operator/(vec const &a, vec const &b) noexcept {
        return zip(a, b, [](T const l, T const r) { return l / r; });
    }
    friend constexpr vec operator%(vec const &a, vec const &b) noexcept {
        static_assert(isIntegralScalar<T>, "operator% requires an integer vector");
        return zip(a, b, [](T const l, T const r) { return l % r; });
    }
    friend constexpr vec operator&(vec const &a, vec const &b) noexcept {
        static_assert(isIntegralScalar<T>, "operator& requires an integer vector");
        return zip(a, b, [](T const l, T const r) { return l & r; });
    }
    friend constexpr vec operator|(vec const &a, vec const &b) noexcept {
        static_assert(isIntegralScalar<T>, "operator| requires an integer vector");
        return zip(a, b, [](T const l, T const r) { return l | r; });
    }
    friend constexpr vec operator^(vec const &a, vec const &b) noexcept {
        static_assert(isIntegralScalar<T>, "operator^ requires an integer vector");
        return zip(a, b, [](T const l, T const r) { return l ^ r; });
    }
    friend constexpr vec operator<<(vec const &a, vec const &b) noexcept {
        static_assert(isIntegralScalar<T>, "operator<< requires an integer vector");
        return zip(a, b, [](T const l, T const r) { return l << r; });
    }
    friend constexpr vec operator>>(vec const &a, vec const &b) noexcept {
        static_assert(isIntegralScalar<T>, "operator>> requires an integer vector");
        return zip(a, b, [](T const l, T const r) { return l >> r; });
    }

    friend constexpr vec operator+(vec const &a, T const s) noexcept { return a + vec{s}; }
    friend constexpr vec operator-(vec const &a, T const s) noexcept { return a - vec{s}; }
    friend constexpr vec operator*(vec const &a, T const s) noexcept { return a * vec{s}; }
    friend constexpr vec operator/(vec const &a, T const s) noexcept { return a / vec{s}; }
    friend constexpr vec operator%(vec const &a, T const s) noexcept { return a % vec{s}; }
    friend constexpr vec operator&(vec const &a, T const s) noexcept { return a & vec{s}; }
    friend constexpr vec operator|(vec const &a, T const s) noexcept { return a | vec{s}; }
    friend constexpr vec operator^(vec const &a, T const s) noexcept { return a ^ vec{s}; }
    friend constexpr vec operator<<(vec const &a, T const s) noexcept { return a << vec{s}; }
    friend constexpr vec operator>>(vec const &a, T const s) noexcept { return a >> vec{s}; }

    friend constexpr vec operator+(T const s, vec const &a) noexcept { return vec{s} + a; }
    friend constexpr vec operator-(T const s, vec const &a) noexcept { return vec{s} - a; }
    friend constexpr vec operator*(T const s, vec const &a) noexcept { return vec{s} * a; }
    friend constexpr vec operator/(T const s, vec const &a) noexcept { return vec{s} / a; }
    friend constexpr vec operator%(T const s, vec const &a) noexcept { return vec{s} % a; }
    friend constexpr vec operator&(T const s, vec const &a) noexcept { return vec{s} & a; }
    friend constexpr vec operator|(T const s, vec const &a) noexcept { return vec{s} | a; }
    friend constexpr vec operator^(T const s, vec const &a) noexcept { return vec{s} ^ a; }

    friend constexpr vec &operator+=(vec &a, vec const &b) noexcept { return a = a + b; }
    friend constexpr vec &operator-=(vec &a, vec const &b) noexcept { return a = a - b; }
    friend constexpr vec &operator*=(vec &a, vec const &b) noexcept { return a = a * b; }
    friend constexpr vec &operator/=(vec &a, vec const &b) noexcept { return a = a / b; }
    friend constexpr vec &operator%=(vec &a, vec const &b) noexcept { return a = a % b; }
    friend constexpr vec &operator&=(vec &a, vec const &b) noexcept { return a = a & b; }
    friend constexpr vec &operator|=(vec &a, vec const &b) noexcept { return a = a | b; }
    friend constexpr vec &operator^=(vec &a, vec const &b) noexcept { return a = a ^ b; }
    friend constexpr vec &operator<<=(vec &a, vec const &b) noexcept { return a = a << b; }
    friend constexpr vec &operator>>=(vec &a, vec const &b) noexcept { return a = a >> b; }

    friend constexpr vec &operator+=(vec &a, T const s) noexcept { return a = a + s; }
    friend constexpr vec &operator-=(vec &a, T const s) noexcept { return a = a - s; }
    friend constexpr vec &operator*=(vec &a, T const s) noexcept { return a = a * s; }
    friend constexpr vec &operator/=(vec &a, T const s) noexcept { return a = a / s; }
    friend constexpr vec &operator%=(vec &a, T const s) noexcept { return a = a % s; }
    friend constexpr vec &operator&=(vec &a, T const s) noexcept { return a = a & s; }
    friend constexpr vec &operator|=(vec &a, T const s) noexcept { return a = a | s; }
    friend constexpr vec &operator^=(vec &a, T const s) noexcept { return a = a ^ s; }
    friend constexpr vec &operator<<=(vec &a, T const s) noexcept { return a = a << s; }
    friend constexpr vec &operator>>=(vec &a, T const s) noexcept { return a = a >> s; }

    friend constexpr vec &operator++(vec &a) noexcept { return a += T{1}; }
    friend constexpr vec &operator--(vec &a) noexcept { return a -= T{1}; }
    friend constexpr vec operator++(vec &a, int) noexcept {
        auto const old = a;
        a += T{1};
        return old;
    }
    friend constexpr vec operator--(vec &a, int) noexcept {
        auto const old = a;
        a -= T{1};
        return old;
    }

    // GLSL `==` on vectors compares all components and yields one bool.
    friend constexpr bool operator==(vec const &a, vec const &b) noexcept {
        for (int i = 0; i < N; ++i) {
            if (a.component(i) != b.component(i)) {
                return false;
            }
        }
        return true;
    }
    friend constexpr bool operator!=(vec const &a, vec const &b) noexcept { return not(a == b); }

private:
    template <class F>
    static constexpr vec map(vec const &a, F const f) noexcept {
        auto result = vec{};
        for (int i = 0; i < N; ++i) {
            result.component(i) = f(a.component(i));
        }
        return result;
    }

    template <class F>
    static constexpr vec zip(vec const &a, vec const &b, F const f) noexcept {
        static_assert(not std::is_same_v<T, bool>, "arithmetic is not defined on bool vectors");
        auto result = vec{};
        for (int i = 0; i < N; ++i) {
            result.component(i) = static_cast<T>(f(a.component(i), b.component(i)));
        }
        return result;
    }
};

using vec2 = vec<float, 2>;
using vec3 = vec<float, 3>;
using vec4 = vec<float, 4>;
using dvec2 = vec<double, 2>;
using dvec3 = vec<double, 3>;
using dvec4 = vec<double, 4>;
using ivec2 = vec<std::int32_t, 2>;
using ivec3 = vec<std::int32_t, 3>;
using ivec4 = vec<std::int32_t, 4>;
using uvec2 = vec<std::uint32_t, 2>;
using uvec3 = vec<std::uint32_t, 3>;
using uvec4 = vec<std::uint32_t, 4>;
using bvec2 = vec<bool, 2>;
using bvec3 = vec<bool, 3>;
using bvec4 = vec<bool, 4>;

// std430 places vec3 on a 16-byte boundary; matching it keeps
// BufferResource<vec3> byte-compatible with a GLSL `vec3 data[]`.
static_assert(sizeof(vec3) == 16 and alignof(vec3) == 16);
static_assert(sizeof(vec2) == 8 and alignof(vec2) == 8);
static_assert(sizeof(vec4) == 16 and alignof(vec4) == 16);

} // namespace tc
