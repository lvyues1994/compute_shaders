#pragma once

#include <cstdint>
#include <cstring>

namespace tc {

namespace detail {

// IEEE 754 binary16 conversions, rounding to nearest even like GPUs.
inline std::uint16_t floatToHalf(float const value) noexcept {
    auto bits = std::uint32_t{};
    std::memcpy(&bits, &value, sizeof bits);
    auto const sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
    auto const exponent = static_cast<std::int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    auto mantissa = bits & 0x7fffffu;
    if (((bits >> 23) & 0xffu) == 0xffu) {
        return static_cast<std::uint16_t>(sign | 0x7c00u | (mantissa != 0 ? 0x200u : 0u));
    }
    if (exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    if (exponent <= 0) {
        if (exponent < -10) {
            return sign;
        }
        mantissa |= 0x800000u;
        auto const shift = static_cast<std::uint32_t>(14 - exponent);
        auto half = mantissa >> shift;
        auto const remainder = mantissa & ((1u << shift) - 1u);
        auto const halfway = 1u << (shift - 1u);
        if (remainder > halfway or (remainder == halfway and (half & 1u) != 0)) {
            ++half;
        }
        return static_cast<std::uint16_t>(sign | half);
    }
    auto half = static_cast<std::uint32_t>(exponent) << 10 | (mantissa >> 13);
    auto const remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u or (remainder == 0x1000u and (half & 1u) != 0)) {
        ++half;
    }
    return static_cast<std::uint16_t>(sign | half);
}

inline float halfToFloat(std::uint16_t const half) noexcept {
    auto const sign = static_cast<std::uint32_t>(half & 0x8000u) << 16;
    auto exponent = static_cast<std::uint32_t>((half >> 10) & 0x1fu);
    auto mantissa = static_cast<std::uint32_t>(half & 0x3ffu);
    auto bits = std::uint32_t{};
    if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else if (exponent != 0) {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    } else if (mantissa != 0) {
        exponent = 127 - 15 + 1;
        while ((mantissa & 0x400u) == 0) {
            mantissa <<= 1;
            --exponent;
        }
        bits = sign | (exponent << 23) | ((mantissa & 0x3ffu) << 13);
    } else {
        bits = sign;
    }
    auto value = 0.0f;
    std::memcpy(&value, &bits, sizeof value);
    return value;
}

} // namespace detail

// A 16-bit float: the storage of gpu::RGBA16F texels and cpu::RGBA16F
// pixels. Kernels compute in float; this type only stores.
struct half {
    std::uint16_t bits = 0;

    half() noexcept = default;
    explicit half(float const value) noexcept : bits(detail::floatToHalf(value)) {}
    operator float() const noexcept { return detail::halfToFloat(bits); }
};

} // namespace tc
