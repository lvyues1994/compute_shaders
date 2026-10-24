#pragma once

#include <cstdint>
#include <string>

#include <tc/device.hpp>
#include <tc/opengl/opengl_device.hpp>

#include "gl_api.hpp"

namespace tc::opengl {

struct GlFormat {
    GLenum internalFormat = GL_NONE;
    GLenum pixelFormat = GL_NONE; // for glTexSubImage*
    GLenum pixelType = GL_NONE;
};

// Throws DeviceError for formats without an OpenGL equivalent.
GlFormat glFormatOf(ImageFormat const &format);

// Compute shader that copies every texel of the texture at unit 0 into the
// SSBO at binding 0 as four 32-bit components (OpenGL ES has no
// glGetTexImage, so both APIs read images back this way). It uses
// texelFetch, which works for formats that cannot be storage images.
std::string readbackShaderSource(ImageFormat const &format, Dim dim, OpenGLApi api);

// Packs four 32-bit shader components (float, int or uint bits) into one
// texel of `format`, using the same rounding as an imageStore.
void encodeTexel(ImageFormat const &format, std::uint32_t const (&components)[4], std::byte *texel);

} // namespace tc::opengl
