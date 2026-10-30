#pragma once

#include <cassert>
#include <cstdint>

#include <tc/buffer.hpp>
#include <tc/device_image.hpp>
#include <tc/dim.hpp>
#include <tc/gpu_format.hpp>
#include <tc/mip_chain.hpp>
#include <tc/pixel.hpp>

namespace tc {

// A storage image binding: `layout(binding = Binding, <G>) uniform image<D>`.
// G is the GPU (internal) format, P the host (external) pixel type of the
// attached BufferResource<P, D>. Upload and download convert between them.
template <class G, Dim D, class P, std::uint32_t Binding>
struct ImageBinding {
    static_assert(D != Dim::D1, "1D images are unavailable on OpenGL ES; use a BufferBinding instead");
    static_assert(isPixel<P>, "P must model the pixel concept (see tc/pixel.hpp)");
    static_assert(isUploadCompatible<G, P>, "integer pixels need an integer GPU format and vice versa");

    using Format = G;
    using Pixel = P;
    using Traits = GPUFormatTraits<G>;
    using Resource = BufferResource<P, D>;
    using Chain = MipChain<P, D>;
    using IndexType = typename DimTraits<D>::IndexType;
    static constexpr Dim dim = D;
    static constexpr std::uint32_t binding = Binding;

    void attach(Resource &resource_) noexcept { target = detail::ImageAttachment<P, D>{&resource_, nullptr, 0}; }
    // Kernels read and write mip level `level` of the chain.
    void attach(Chain &chain_, std::uint32_t const level = 0) noexcept {
        assert(level < chain_.levels());
        target = detail::ImageAttachment<P, D>{nullptr, &chain_, level};
    }
    void attach(DeviceImage<G, D, P> const &image, std::uint32_t const level = 0) noexcept {
        assert(image and level < image.levels());
        target = detail::ImageAttachment<P, D>{nullptr, nullptr, level, image.shared()};
    }

    // The level kernels access; nullptr for a device image on a GPU device.
    Resource *attachedResource() const noexcept { return target.resource(target.level); }
    bool isDeviceImage() const noexcept { return target.device != nullptr; }
    std::uint32_t level() const noexcept { return target.level; }
    // The whole attachment, for transfers.
    std::uint32_t levelCount() const noexcept { return target.levels(); }
    Resource *levelResource(std::uint32_t const index) const noexcept { return target.resource(index); }
    ResourceIdentity const *identity() const noexcept { return target.identity(); }

private:
    detail::ImageAttachment<P, D> target;
};

template <class I>
struct IsImageBinding : std::false_type {};

template <class G, Dim D, class P, std::uint32_t B>
struct IsImageBinding<ImageBinding<G, D, P, B>> : std::true_type {};

// GLSL imageSize: int-vector extent of the bound image (level).
template <class G, Dim D, class P, std::uint32_t B>
typename DimTraits<D>::IndexType imageSize(ImageBinding<G, D, P, B> const &image) noexcept {
    assert(image.attachedResource() != nullptr and "ImageBinding used before attach()");
    return image.attachedResource()->extent();
}

// GLSL imageLoad. The host pixel is converted to the GPU storage format and
// decoded exactly as the GPU would, so CPU and GPU results agree. Loads
// outside the image return zero, as in OpenGL.
template <class G, Dim D, class P, std::uint32_t B>
typename GPUFormatTraits<G>::VectorType imageLoad(ImageBinding<G, D, P, B> const &image,
                                                  typename DimTraits<D>::IndexType const coord) noexcept {
    using Traits = GPUFormatTraits<G>;
    auto const *resource = image.attachedResource();
    assert(resource != nullptr and "ImageBinding used before attach()");
    if (not DimTraits<D>::contains(coord, resource->extent())) {
        return typename Traits::VectorType{};
    }
    auto const stored = convertPixel<typename Traits::StoragePixel>((*resource)[coord]);
    return decodeTexel<G>(stored);
}

// GLSL imageStore. The value is encoded into the GPU storage format, then
// converted to the host pixel type. Stores outside the image are ignored.
template <class G, Dim D, class P, std::uint32_t B>
void imageStore(ImageBinding<G, D, P, B> const &image, typename DimTraits<D>::IndexType const coord,
                typename GPUFormatTraits<G>::VectorType const &value) noexcept {
    auto *resource = image.attachedResource();
    assert(resource != nullptr and "ImageBinding used before attach()");
    if (not DimTraits<D>::contains(coord, resource->extent())) {
        return;
    }
    (*resource)[coord] = convertPixel<P>(encodeTexel<G>(value));
}

} // namespace tc
