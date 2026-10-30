#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include <tc/glsl_math.hpp>
#include <tc/half.hpp>
#include <tc/vec.hpp>

// GLSL packing and bit-manipulation built-ins (all part of OpenGL ES 3.1),
// the tools for reading 8- and 16-bit pixel data stored in uint buffers.
namespace tc {

namespace detail {

inline std::uint32_t quantize(float const value, float const lo, float const hi, float const scale) noexcept {
    auto const clamped = value < lo ? lo : (value > hi ? hi : value);
    return static_cast<std::uint32_t>(static_cast<std::int32_t>(std::nearbyint(clamped * scale)));
}

template <class V>
using EnableGenInteger = std::enable_if_t<isIntegralScalar<ComponentOf<V>>, int>;

template <class V>
using IntResult = typename VecTraits<V>::template Rebind<std::int32_t>;

} // namespace detail

inline uint packUnorm4x8(vec4 const &v) noexcept {
    auto packed = uint{0};
    for (int i = 0; i < 4; ++i) {
        packed |= detail::quantize(v[i], 0.0f, 1.0f, 255.0f) << (8 * i);
    }
    return packed;
}

inline vec4 unpackUnorm4x8(uint const p) noexcept {
    auto v = vec4{};
    for (int i = 0; i < 4; ++i) {
        v[i] = static_cast<float>((p >> (8 * i)) & 0xffu) / 255.0f;
    }
    return v;
}

inline uint packSnorm4x8(vec4 const &v) noexcept {
    auto packed = uint{0};
    for (int i = 0; i < 4; ++i) {
        packed |= (detail::quantize(v[i], -1.0f, 1.0f, 127.0f) & 0xffu) << (8 * i);
    }
    return packed;
}

inline vec4 unpackSnorm4x8(uint const p) noexcept {
    auto v = vec4{};
    for (int i = 0; i < 4; ++i) {
        auto const byte = static_cast<std::int8_t>(static_cast<std::uint8_t>((p >> (8 * i)) & 0xffu));
        v[i] = clamp(static_cast<float>(byte) / 127.0f, -1.0f, 1.0f);
    }
    return v;
}

inline uint packUnorm2x16(vec2 const &v) noexcept {
    return detail::quantize(v.x, 0.0f, 1.0f, 65535.0f) | (detail::quantize(v.y, 0.0f, 1.0f, 65535.0f) << 16);
}

inline vec2 unpackUnorm2x16(uint const p) noexcept {
    return vec2{static_cast<float>(p & 0xffffu) / 65535.0f, static_cast<float>(p >> 16) / 65535.0f};
}

inline uint packSnorm2x16(vec2 const &v) noexcept {
    return (detail::quantize(v.x, -1.0f, 1.0f, 32767.0f) & 0xffffu)
         | ((detail::quantize(v.y, -1.0f, 1.0f, 32767.0f) & 0xffffu) << 16);
}

inline vec2 unpackSnorm2x16(uint const p) noexcept {
    auto const lo = static_cast<std::int16_t>(static_cast<std::uint16_t>(p & 0xffffu));
    auto const hi = static_cast<std::int16_t>(static_cast<std::uint16_t>(p >> 16));
    return vec2{clamp(static_cast<float>(lo) / 32767.0f, -1.0f, 1.0f), clamp(static_cast<float>(hi) / 32767.0f, -1.0f, 1.0f)};
}

inline uint packHalf2x16(vec2 const &v) noexcept {
    return uint{detail::floatToHalf(v.x)} | (uint{detail::floatToHalf(v.y)} << 16);
}

inline vec2 unpackHalf2x16(uint const p) noexcept {
    return vec2{detail::halfToFloat(static_cast<std::uint16_t>(p & 0xffffu)),
                detail::halfToFloat(static_cast<std::uint16_t>(p >> 16))};
}

// bitfieldExtract: `bits` bits starting at `offset`; signed results are
// sign-extended from the top extracted bit.
template <class V, detail::EnableGenInteger<V> = 0>
constexpr V bitfieldExtract(V const &value, std::int32_t const offset, std::int32_t const bits) noexcept {
    return detail::componentwise(value, [offset, bits](auto const c) {
        using C = std::remove_const_t<decltype(c)>;
        if (bits == 0) {
            return C{0};
        }
        auto const raw = static_cast<std::uint32_t>(c) >> offset;
        auto const mask = bits == 32 ? 0xffffffffu : (1u << bits) - 1u;
        auto field = raw & mask;
        if constexpr (std::is_signed_v<C>) {
            if (bits < 32 and (field & (1u << (bits - 1))) != 0) {
                field |= ~mask;
            }
        }
        return static_cast<C>(field);
    });
}

template <class V, detail::EnableGenInteger<V> = 0>
constexpr V bitfieldInsert(V const &base, V const &insert, std::int32_t const offset, std::int32_t const bits) noexcept {
    return detail::componentwise(base, insert, [offset, bits](auto const b, auto const i) {
        using C = std::remove_const_t<decltype(b)>;
        if (bits == 0) {
            return b;
        }
        auto const mask = (bits == 32 ? 0xffffffffu : (1u << bits) - 1u) << offset;
        auto const merged = (static_cast<std::uint32_t>(b) & ~mask) | ((static_cast<std::uint32_t>(i) << offset) & mask);
        return static_cast<C>(merged);
    });
}

template <class V, detail::EnableGenInteger<V> = 0>
constexpr V bitfieldReverse(V const &value) noexcept {
    return detail::componentwise(value, [](auto const c) {
        using C = std::remove_const_t<decltype(c)>;
        auto bits = static_cast<std::uint32_t>(c);
        auto reversed = std::uint32_t{0};
        for (int i = 0; i < 32; ++i) {
            reversed = (reversed << 1) | (bits & 1u);
            bits >>= 1;
        }
        return static_cast<C>(reversed);
    });
}

template <class V, detail::EnableGenInteger<V> = 0>
constexpr detail::IntResult<V> bitCount(V const &value) noexcept {
    return detail::componentwise(value, [](auto const c) {
        auto bits = static_cast<std::uint32_t>(c);
        auto count = std::int32_t{0};
        while (bits != 0) {
            count += static_cast<std::int32_t>(bits & 1u);
            bits >>= 1;
        }
        return count;
    });
}

// Index of the lowest set bit, -1 for zero.
template <class V, detail::EnableGenInteger<V> = 0>
constexpr detail::IntResult<V> findLSB(V const &value) noexcept {
    return detail::componentwise(value, [](auto const c) {
        auto const bits = static_cast<std::uint32_t>(c);
        for (std::int32_t i = 0; i < 32; ++i) {
            if ((bits >> i) & 1u) {
                return i;
            }
        }
        return std::int32_t{-1};
    });
}

// Index of the highest set bit; for negative signed values the highest
// clear bit. -1 for 0 (and for -1 when signed).
template <class V, detail::EnableGenInteger<V> = 0>
constexpr detail::IntResult<V> findMSB(V const &value) noexcept {
    return detail::componentwise(value, [](auto const c) {
        using C = std::remove_const_t<decltype(c)>;
        auto bits = static_cast<std::uint32_t>(c);
        if constexpr (std::is_signed_v<C>) {
            if (c < 0) {
                bits = ~bits;
            }
        }
        for (std::int32_t i = 31; i >= 0; --i) {
            if ((bits >> i) & 1u) {
                return i;
            }
        }
        return std::int32_t{-1};
    });
}

} // namespace tc
