#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include <tc/buffer.hpp>
#include <tc/dim.hpp>
#include <tc/gpu_format.hpp>
#include <tc/mip_chain.hpp>
#include <tc/pixel.hpp>

namespace tc {

namespace detail {

// What the copies of one DeviceImage share.
template <class P, Dim D>
struct DeviceImageState {
    DeviceImageState(typename DimTraits<D>::IndexType const extent_, std::uint32_t const levels_)
        : extent(extent_), levels(levels_), identity(ResourceIdentity::create()) {}

    typename DimTraits<D>::IndexType extent;
    std::uint32_t levels;
    ResourceIdentity identity;
    // The image itself on the CPU backend, which runs kernels on the host;
    // empty on GPU devices.
    std::unique_ptr<MipChain<P, D>> host;
};

} // namespace detail

// An image with mip levels that lives only on the device that created it
// (Device::createImage), like a buffer from createBuffer() but without a
// host mapping: there is no host copy, so it occupies device memory only.
// Kernels write it through ImageBinding (one level) and sample it through
// TextureBinding; Device::copyBufferToImage fills it from a buffer on the
// device. uploadImage() and downloadImage() do not apply. The contents
// start undefined. Copies refer to the same image, which is freed after the
// last copy, once the device no longer uses it.
//
// G is the GPU format, P the pixel type of the bindings it attaches to (the
// CPU backend stores P).
template <class G, Dim D, class P = typename GPUFormatTraits<G>::StoragePixel>
struct DeviceImage {
    static_assert(isPixel<P>, "P must model the pixel concept (see tc/pixel.hpp)");
    static_assert(isUploadCompatible<G, P>, "integer pixels need an integer GPU format and vice versa");

    using Format = G;
    using Pixel = P;
    using IndexType = typename DimTraits<D>::IndexType;
    static constexpr Dim dim = D;

    DeviceImage() = default; // no image

    explicit operator bool() const noexcept { return state != nullptr; }
    // Array textures: layers in z, the same on every level.
    IndexType extent(std::uint32_t const level = 0) const noexcept {
        return DimTraits<D>::fromExtent3(mipExtent(D, DimTraits<D>::toExtent3(state->extent), level));
    }
    std::uint32_t levels() const noexcept { return state->levels; }
    ResourceIdentity const &identity() const noexcept { return state->identity; }

    detail::DeviceImageState<P, D> const *shared() const noexcept { return state.get(); }

private:
    friend struct Device;
    explicit DeviceImage(std::shared_ptr<detail::DeviceImageState<P, D>> state_) noexcept : state(std::move(state_)) {}

    std::shared_ptr<detail::DeviceImageState<P, D>> state;
};

namespace detail {

// What an image or texture binding is attached to: one resource (a single
// level), a mip chain or a device image, and for images the level kernels
// access.
template <class P, Dim D>
struct ImageAttachment {
    BufferResource<P, D> *single = nullptr;
    MipChain<P, D> *chain = nullptr;
    std::uint32_t level = 0;
    DeviceImageState<P, D> const *device = nullptr;

    bool attached() const noexcept { return single != nullptr or chain != nullptr or device != nullptr; }
    std::uint32_t levels() const noexcept {
        return chain != nullptr ? chain->levels() : device != nullptr ? device->levels : 1u;
    }
    // Host storage of a level; nullptr for device images on GPU devices.
    BufferResource<P, D> *resource(std::uint32_t const index) const noexcept {
        if (device != nullptr) {
            return device->host != nullptr and index < device->host->levels() ? &device->host->level(index) : nullptr;
        }
        if (chain != nullptr) {
            return index < chain->levels() ? &chain->level(index) : nullptr;
        }
        return index == 0 ? single : nullptr;
    }
    ResourceIdentity const *identity() const noexcept {
        return device != nullptr  ? &device->identity
             : chain != nullptr   ? &chain->identity()
             : single != nullptr ? &single->identity()
                                  : nullptr;
    }
};

} // namespace detail

} // namespace tc
