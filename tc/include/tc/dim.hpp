#pragma once

#include <algorithm>
#include <cstdint>

#include <tc/vec.hpp>

namespace tc {

enum class Dim : std::uint8_t {
    D1 = 1,
    D2 = 2,
    D3 = 3,
    D2Array = 4, // 2D layers: coordinates (x, y, layer)
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
    static constexpr IndexType fromExtent3(ivec3 const size) noexcept { return size.x; }
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
    static constexpr IndexType fromExtent3(ivec3 const size) noexcept { return ivec2{size.x, size.y}; }
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
    static constexpr IndexType fromExtent3(ivec3 const size) noexcept { return size; }
};

// Layers are stored one after another, like the z slices of a 3D image.
template <>
struct DimTraits<Dim::D2Array> : DimTraits<Dim::D3> {};

template <Dim D>
using tcVec = typename DimTraits<D>::IndexType;

// Extent of mip level `level` of an image whose level 0 is `extent` (x, y, z
// or layers): every axis halves, rounding down to 1, except array layers.
constexpr ivec3 mipExtent(Dim const dim, ivec3 const extent, std::uint32_t const level) noexcept {
    auto const halve = [level](std::int32_t const size) { return std::max(size >> level, 1); };
    return ivec3{halve(extent.x), dim == Dim::D1 ? 1 : halve(extent.y),
                 dim == Dim::D3 ? halve(extent.z) : extent.z};
}

// Levels of a full mip chain down to 1 x 1 (x 1).
constexpr std::uint32_t fullMipLevels(Dim const dim, ivec3 const extent) noexcept {
    auto largest = std::max(extent.x, dim == Dim::D1 ? 1 : extent.y);
    largest = std::max(largest, dim == Dim::D3 ? extent.z : 1);
    auto levels = std::uint32_t{1};
    while (largest > 1) {
        largest >>= 1;
        ++levels;
    }
    return levels;
}

} // namespace tc
