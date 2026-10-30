#pragma once

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>

#include <tc/buffer.hpp>
#include <tc/dim.hpp>
#include <tc/gpu_format.hpp>
#include <tc/pixel.hpp>
#include <tc/texture.hpp>
#include <tc/vec.hpp>

#if defined(__ANDROID__)
#include <android/hardware_buffer.h>
#endif

namespace tc {

// A frame owned by the platform and shared without copies: on Android an
// AHardwareBuffer from the camera, a video decoder, an ImageReader, or one
// the application allocated to present. Kernels sample it through
// ExternalTextureBinding (YUV frames read as RGB) and write RGBA frames
// through ExternalImageBinding. Copies refer to the same frame.
struct ExternalImage {
#if defined(__ANDROID__)
    // Takes its own reference on `buffer`.
    static ExternalImage fromHardwareBuffer(AHardwareBuffer *buffer);
    AHardwareBuffer *hardwareBuffer() const noexcept { return static_cast<AHardwareBuffer *>(native.get()); }
#endif

    ivec2 size() const noexcept { return extent; }
    ResourceIdentity const &identity() const noexcept { return id; }

    // The storage image format of an RGBA frame (rgba8 or rgba16f); nullopt
    // for YUV and other formats kernels can only sample.
    std::optional<ImageFormat> storageFormat() const noexcept { return format; }

    // The CPU backend's copy of the frame: snapshot() decodes the current
    // contents into it (YUV as ITU-R BT.601 limited range, like the Vulkan
    // backend's sampler conversion); writeBack() stores it into the frame.
    void snapshot();
    void writeBack();
    BufferResource<cpu::RGBA32F, Dim::D2> *cpuFrame() const noexcept { return frame.get(); }

private:
    ExternalImage(std::shared_ptr<void> native_, ivec2 const extent_, std::optional<ImageFormat> const format_)
        : native(std::move(native_)), extent(extent_), format(format_), id(ResourceIdentity::create()) {}

    std::shared_ptr<void> native;
    ivec2 extent{};
    std::optional<ImageFormat> format;
    ResourceIdentity id;
    std::shared_ptr<BufferResource<cpu::RGBA32F, Dim::D2>> frame;
};

// An external frame in a kernel: `uniform samplerExternalOES` on OpenGL ES,
// a sampler with YCbCr conversion on Vulkan. Sampled with bilinear
// filtering and clamp-to-edge addressing.
template <std::uint32_t Binding>
struct ExternalTextureBinding {
    static constexpr std::uint32_t binding = Binding;

    void attach(ExternalImage &image_) noexcept { image = &image_; }
    ExternalImage *attachedImage() const noexcept { return image; }

private:
    ExternalImage *image = nullptr;
};

template <std::uint32_t B>
ivec2 textureSize(ExternalTextureBinding<B> const &texture, std::int32_t const lod) noexcept {
    assert(texture.attachedImage() != nullptr and "ExternalTextureBinding used before attach()");
    assert(lod == 0);
    (void)lod;
    return texture.attachedImage()->size();
}

template <std::uint32_t B>
vec4 texture(ExternalTextureBinding<B> const &texture, vec2 const &uv) noexcept {
    assert(texture.attachedImage() != nullptr and "ExternalTextureBinding used before attach()");
    auto *frame = texture.attachedImage()->cpuFrame();
    assert(frame != nullptr and "bind the external texture on the device before executing");
    auto sampler = TextureBinding<gpu::RGBA32F, Dim::D2, cpu::RGBA32F, B, Filter::Linear, Wrap::ClampToEdge>{};
    sampler.attach(*frame);
    return tc::texture(sampler, uv);
}

// An RGBA platform frame as a storage image, `layout(G) uniform image2D`:
// kernels write it in place, e.g. render into a buffer the compositor then
// shows. G must match the frame (gpu::RGBA8 or gpu::RGBA16F). Allocate it
// with GPU sampling and framebuffer usage; GPU backends report a
// DeviceError when the driver cannot write it.
template <class G, std::uint32_t Binding>
struct ExternalImageBinding {
    static_assert(G::encoding != Encoding::Integer, "platform frames are normalized or float");

    using Format = G;
    using Traits = GPUFormatTraits<G>;
    static constexpr Dim dim = Dim::D2;
    static constexpr std::uint32_t binding = Binding;

    void attach(ExternalImage &image_) noexcept { image = &image_; }
    ExternalImage *attachedImage() const noexcept { return image; }

private:
    ExternalImage *image = nullptr;
};

namespace detail {

inline BufferResource<cpu::RGBA32F, Dim::D2> &cpuFrameOf(ExternalImage const *image) noexcept {
    assert(image != nullptr and "ExternalImageBinding used before attach()");
    assert(image->cpuFrame() != nullptr and "bind the external image on the device before executing");
    return *image->cpuFrame();
}

} // namespace detail

template <class G, std::uint32_t B>
ivec2 imageSize(ExternalImageBinding<G, B> const &image) noexcept {
    return detail::cpuFrameOf(image.attachedImage()).extent();
}

// Loads and stores go through G's encoding, as on the GPU.
template <class G, std::uint32_t B>
vec4 imageLoad(ExternalImageBinding<G, B> const &image, ivec2 const coord) noexcept {
    auto const &frame = detail::cpuFrameOf(image.attachedImage());
    if (not DimTraits<Dim::D2>::contains(coord, frame.extent())) {
        return vec4{};
    }
    return decodeTexel<G>(convertPixel<typename GPUFormatTraits<G>::StoragePixel>(frame[coord]));
}

template <class G, std::uint32_t B>
void imageStore(ExternalImageBinding<G, B> const &image, ivec2 const coord, vec4 const &value) noexcept {
    auto &frame = detail::cpuFrameOf(image.attachedImage());
    if (DimTraits<Dim::D2>::contains(coord, frame.extent())) {
        frame[coord] = convertPixel<cpu::RGBA32F>(encodeTexel<G>(value));
    }
}

} // namespace tc
