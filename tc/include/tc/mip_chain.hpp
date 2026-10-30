#pragma once

#include <cassert>
#include <cstdint>
#include <vector>

#include <tc/buffer.hpp>
#include <tc/dim.hpp>

namespace tc {

// An image with mip levels: level 0 has `extent`, each further level halves
// x and y (and z of Dim::D3; array layers stay), rounding down to 1. Texture
// bindings sample the whole chain; image bindings write one level, e.g. to
// generate the chain with a kernel. A copy is a new resource.
template <class P, Dim D>
struct MipChain {
    using Level = BufferResource<P, D>;
    using IndexType = typename DimTraits<D>::IndexType;

    // `levels` == 0: the full chain down to one texel.
    explicit MipChain(IndexType const extent, std::uint32_t levels = 0) : id(ResourceIdentity::create()) {
        auto const extent3 = DimTraits<D>::toExtent3(extent);
        auto const full = fullMipLevels(D, extent3);
        levels = levels == 0 ? full : levels;
        assert(levels <= full and "more mip levels than the extent allows");
        chain.reserve(levels);
        for (std::uint32_t level = 0; level < levels; ++level) {
            chain.emplace_back(DimTraits<D>::fromExtent3(mipExtent(D, extent3, level)));
        }
    }

    MipChain(MipChain const &other) : chain(other.chain), id(ResourceIdentity::create()) {}
    MipChain &operator=(MipChain const &other) {
        if (this != &other) {
            chain = other.chain;
            id = ResourceIdentity::create();
        }
        return *this;
    }
    MipChain(MipChain &&) noexcept = default;
    MipChain &operator=(MipChain &&) noexcept = default;
    ~MipChain() = default;

    std::uint32_t levels() const noexcept { return static_cast<std::uint32_t>(chain.size()); }
    Level &level(std::uint32_t const index) noexcept {
        assert(index < chain.size());
        return chain[index];
    }
    Level const &level(std::uint32_t const index) const noexcept {
        assert(index < chain.size());
        return chain[index];
    }
    IndexType extent(std::uint32_t const index = 0) const noexcept { return level(index).extent(); }

    // Identifies the whole chain to the GPU backends.
    ResourceIdentity const &identity() const noexcept { return id; }

private:
    std::vector<Level> chain;
    ResourceIdentity id;
};

} // namespace tc
