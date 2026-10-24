#pragma once

#include <cassert>
#include <cstdint>
#include <memory>

#include <tc/buffer.hpp>
#include <tc/dim.hpp>
#include <tc/pixel.hpp>
#include <tc/texture.hpp>
#include <tc/vec.hpp>

#if defined(__ANDROID__)
#include <android/hardware_buffer.h>
#endif

namespace tc {

// A frame owned by the platform and shared without copies: on Android an
// AHardwareBuffer from the camera, a video decoder or an ImageReader.
// Kernels sample it through ExternalTextureBinding; YUV frames read as RGB.
// Copies refer to the same frame.
struct ExternalImage {
#if defined(__ANDROID__)
    // Takes its own reference on `buffer`.
    static ExternalImage fromHardwareBuffer(AHardwareBuffer *buffer);
    AHardwareBuffer *hardwareBuffer() const noexcept { return static_cast<AHardwareBuffer *>(native.get()); }
#endif

    ivec2 size() const noexcept { return extent; }
    ResourceIdentity const &identity() const noexcept { return id; }

    // Decodes the current frame contents into the RGBA copy the CPU backend
    // samples. YUV frames are decoded as ITU-R BT.601 limited range, like
    // the Vulkan backend's sampler conversion.
    void snapshot();
    BufferResource<cpu::RGBA32F, Dim::D2> *cpuFrame() const noexcept { return frame.get(); }

private:
    ExternalImage(std::shared_ptr<void> native_, ivec2 const extent_)
        : native(std::move(native_)), extent(extent_), id(ResourceIdentity::create()) {}

    std::shared_ptr<void> native;
    ivec2 extent{};
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

} // namespace tc
