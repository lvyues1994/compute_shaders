#pragma once

#include <array>
#include <cstdint>
#include <type_traits>

#include <tc/glsl_math.hpp>
#include <tc/glsl_packing.hpp>
#include <tc/mat.hpp>
#include <tc/vec.hpp>

namespace tc {

// A table field stored through a codec. In kernels and host code it is a V
// (`.value`, or by implicit conversion); in a table it takes Codec::words
// words. A codec is a struct written in kernel C++, so tc-transpile turns
// encode and decode into GLSL like any other function:
//
//   struct MyCodec {
//       using Value = V;
//       static constexpr std::uint32_t words = N;
//       static std::array<uint, N> encode(V const &value);
//       static V decode(std::array<uint, N> const &packed);
//   };
//
// Only tables apply the codec; locals and std430 buffers hold the plain V.
template <class V, class Codec>
struct Encoded {
    static_assert(std::is_same_v<V, typename Codec::Value>, "Encoded<V, Codec> needs Codec::Value to be V");
    using value_type = V;
    using codec_type = Codec;

    V value{};

    constexpr Encoded() noexcept = default;
    constexpr Encoded(V const &value_) noexcept : value(value_) {}

    constexpr operator V &() noexcept { return value; }
    constexpr operator V const &() const noexcept { return value; }

    constexpr Encoded &operator=(V const &value_) noexcept {
        value = value_;
        return *this;
    }
};

// Four channels in [0, 1] at 8 bits: colours, 1 word instead of 4.
struct Unorm8x4 {
    using Value = vec4;
    static constexpr std::uint32_t words = 1;
    static std::array<uint, 1> encode(vec4 const &value) { return std::array<uint, 1>{packUnorm4x8(value)}; }
    static vec4 decode(std::array<uint, 1> const &packed) { return unpackUnorm4x8(packed[0]); }
};

// Four channels in [-1, 1] at 8 bits.
struct Snorm8x4 {
    using Value = vec4;
    static constexpr std::uint32_t words = 1;
    static std::array<uint, 1> encode(vec4 const &value) { return std::array<uint, 1>{packSnorm4x8(value)}; }
    static vec4 decode(std::array<uint, 1> const &packed) { return unpackSnorm4x8(packed[0]); }
};

// Two channels in [0, 1] at 16 bits: texture coordinates.
struct Unorm16x2 {
    using Value = vec2;
    static constexpr std::uint32_t words = 1;
    static std::array<uint, 1> encode(vec2 const &value) { return std::array<uint, 1>{packUnorm2x16(value)}; }
    static vec2 decode(std::array<uint, 1> const &packed) { return unpackUnorm2x16(packed[0]); }
};

// Half-precision floats: 1 word instead of 2.
struct Half2 {
    using Value = vec2;
    static constexpr std::uint32_t words = 1;
    static std::array<uint, 1> encode(vec2 const &value) { return std::array<uint, 1>{packHalf2x16(value)}; }
    static vec2 decode(std::array<uint, 1> const &packed) { return unpackHalf2x16(packed[0]); }
};

// Half-precision floats: 2 words instead of 4.
struct Half4 {
    using Value = vec4;
    static constexpr std::uint32_t words = 2;
    static std::array<uint, 2> encode(vec4 const &value) {
        return std::array<uint, 2>{packHalf2x16(value.xy()), packHalf2x16(value.zw())};
    }
    static vec4 decode(std::array<uint, 2> const &packed) {
        return vec4(unpackHalf2x16(packed[0]), unpackHalf2x16(packed[1]));
    }
};

// An affine transform: the last row of the matrix is (0, 0, 0, 1) and is not
// stored, 12 words instead of 16.
struct Affine3x4 {
    using Value = mat4;
    static constexpr std::uint32_t words = 12;
    static std::array<uint, 12> encode(mat4 const &value) {
        std::array<uint, 12> packed{};
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 3; ++r) {
                packed[c * 3 + r] = floatBitsToUint(value[c][r]);
            }
        }
        return packed;
    }
    static mat4 decode(std::array<uint, 12> const &packed) {
        mat4 value = mat4(1.0f);
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 3; ++r) {
                value[c][r] = uintBitsToFloat(packed[c * 3 + r]);
            }
        }
        return value;
    }
};

// A unit vector in octahedral encoding, two 16-bit snorm coordinates: 1
// word instead of 3, error below 0.0001 radians.
struct OctahedralNormal {
    using Value = vec3;
    static constexpr std::uint32_t words = 1;
    static std::array<uint, 1> encode(vec3 const &value) {
        vec3 n = value / (abs(value.x) + abs(value.y) + abs(value.z));
        vec2 p = n.xy();
        if (n.z < 0.0f) {
            vec2 flipped = vec2(1.0f) - abs(vec2{p.y, p.x});
            p = vec2{p.x >= 0.0f ? flipped.x : -flipped.x, p.y >= 0.0f ? flipped.y : -flipped.y};
        }
        return std::array<uint, 1>{packSnorm2x16(p)};
    }
    static vec3 decode(std::array<uint, 1> const &packed) {
        vec2 p = unpackSnorm2x16(packed[0]);
        vec3 n = vec3{p.x, p.y, 1.0f - abs(p.x) - abs(p.y)};
        float t = max(-n.z, 0.0f);
        n.x += n.x >= 0.0f ? -t : t;
        n.y += n.y >= 0.0f ? -t : t;
        return normalize(n);
    }
};

} // namespace tc
