#pragma once

#include <cstdint>
#include <memory>

#include <tc/external_image.hpp>

#include "egl_context.hpp"

namespace tc::opengl {

inline constexpr GLenum kTextureExternalOes = 0x8D65;

// A platform frame imported as a GL_TEXTURE_EXTERNAL_OES texture through an
// EGLImage, sampled by `samplerExternalOES` (OpenGL ES on Android).
struct ExternalTexture {
    virtual ~ExternalTexture() = default;
    virtual GLuint texture() const noexcept = 0;
    // Platform pixel format, e.g. AHARDWAREBUFFER_FORMAT_*.
    virtual std::uint32_t format() const noexcept = 0;
};

// Throws DeviceError when the platform or driver cannot import `image`.
std::unique_ptr<ExternalTexture> importExternalTexture(EglContext const &egl, ExternalImage const &image);

// An RGBA frame as an immutable GL_TEXTURE_2D that image units can write
// (GL_EXT_EGL_image_storage).
std::unique_ptr<ExternalTexture> importExternalStorage(EglContext const &egl, ExternalImage const &image);

} // namespace tc::opengl
