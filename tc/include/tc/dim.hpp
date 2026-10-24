#pragma once

#include <cstdint>

#include <tc/vec.hpp>

namespace tc {

enum class Dim : std::uint8_t {
    D1 = 1,
    D2 = 2,
    D3 = 3,
};

// Maps an n-dimensional coordinate onto the flat storage of a resource. The
// index types are signed because GLSL addresses images with int/ivec.
template <Dim D>
struct DimTraits;

template <>
struct DimTraits<Dim::D1> {
    using IndexType = std::int32_t;

    static constexpr std::int32_t flatten(IndexType const coord, IndexType /*size*/) noexcept { return coord; }
    static constexpr std::int32_t product(IndexType const size) noexcept { return size; }
    static constexpr bool contains(IndexType const coord, IndexType const size) noexcept {
        return coord >= 0 and coord < size;
    }
    static constexpr ivec3 toExtent3(IndexType const size) noexcept { return ivec3{size, 1, 1}; }
};

template <>
struct DimTraits<Dim::D2> {
    using IndexType = ivec2;

    static constexpr std::int32_t flatten(IndexType const coord, IndexType const size) noexcept {
        return coord.y * size.x + coord.x;
    }
    static constexpr std::int32_t product(IndexType const size) noexcept { return size.x * size.y; }
    static constexpr bool contains(IndexType const coord, IndexType const size) noexcept {
        return coord.x >= 0 and coord.y >= 0 and coord.x < size.x and coord.y < size.y;
    }
    static constexpr ivec3 toExtent3(IndexType const size) noexcept { return ivec3{size, 1}; }
};

template <>
struct DimTraits<Dim::D3> {
    using IndexType = ivec3;

    static constexpr std::int32_t flatten(IndexType const coord, IndexType const size) noexcept {
        return (coord.z * size.y + coord.y) * size.x + coord.x;
    }
    static constexpr std::int32_t product(IndexType const size) noexcept { return size.x * size.y * size.z; }
    static constexpr bool contains(IndexType const coord, IndexType const size) noexcept {
        return coord.x >= 0 and coord.y >= 0 and coord.z >= 0 and coord.x < size.x and coord.y < size.y
            and coord.z < size.z;
    }
    static constexpr ivec3 toExtent3(IndexType const size) noexcept { return size; }
};

template <Dim D>
using tcVec = typename DimTraits<D>::IndexType;

} // namespace tc
