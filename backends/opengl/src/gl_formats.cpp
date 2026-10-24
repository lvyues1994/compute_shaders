#include "gl_formats.hpp"

#include <cmath>
#include <cstring>

namespace tc::opengl {

namespace {

GLenum pixelFormatOf(ImageFormat const &format) {
    auto const isInteger = format.encoding == Encoding::Integer;
    switch (format.channelCount) {
    case 1:
        return isInteger ? GL_RED_INTEGER : GL_RED;
    case 2:
        return isInteger ? GL_RG_INTEGER : GL_RG;
    case 4:
        return isInteger ? GL_RGBA_INTEGER : GL_RGBA;
    default:
        return GL_NONE;
    }
}

GLenum pixelTypeOf(ImageFormat const &format) {
    if (format.encoding == Encoding::Float) {
        return format.bitsPerChannel == 32 ? GL_FLOAT : GL_NONE;
    }
    switch (format.bitsPerChannel) {
    case 8:
        return format.isSigned ? GL_BYTE : GL_UNSIGNED_BYTE;
    case 16:
        return format.isSigned ? GL_SHORT : GL_UNSIGNED_SHORT;
    case 32:
        return format.isSigned ? GL_INT : GL_UNSIGNED_INT;
    default:
        return GL_NONE;
    }
}

GLenum internalFormatOf(ImageFormat const &format) {
    auto const q = glslFormatQualifier(format);
    struct Entry {
        char const *qualifier;
        GLenum internal;
    };
    static constexpr Entry table[] = {
        {"r8", GL_R8},           {"rg8", GL_RG8},         {"rgba8", GL_RGBA8},       {"r8ui", GL_R8UI},
        {"rgba8ui", GL_RGBA8UI}, {"rgba8i", GL_RGBA8I},   {"r32ui", GL_R32UI},       {"rg32ui", GL_RG32UI},
        {"rgba32ui", GL_RGBA32UI}, {"r32i", GL_R32I},     {"rgba32i", GL_RGBA32I},   {"r32f", GL_R32F},
        {"rg32f", GL_RG32F},     {"rgba32f", GL_RGBA32F},
    };
    for (auto const &entry : table) {
        if (q == entry.qualifier) {
            return entry.internal;
        }
    }
    return GL_NONE;
}

} // namespace

GlFormat glFormatOf(ImageFormat const &format) {
    auto const result = GlFormat{internalFormatOf(format), pixelFormatOf(format), pixelTypeOf(format)};
    if (result.internalFormat == GL_NONE or result.pixelFormat == GL_NONE or result.pixelType == GL_NONE) {
        throw DeviceError{"image format " + glslFormatQualifier(format) + " is not supported by the OpenGL backend"};
    }
    return result;
}

std::string readbackShaderSource(ImageFormat const &format, Dim const dim, OpenGLApi const api) {
    auto const prefix = format.encoding != Encoding::Integer ? "" : format.isSigned ? "i" : "u";
    auto const scalar = format.encoding != Encoding::Integer ? "float" : format.isSigned ? "int" : "uint";
    auto const is3d = dim == Dim::D3;
    auto source = std::string{api == OpenGLApi::ES ? "#version 310 es\nprecision highp float;\nprecision highp int;\n"
                                                   : "#version 430\n"};
    source += "layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;\n";
    source += "layout(binding = 0, " + glslFormatQualifier(format) + ") readonly uniform highp " + prefix + "image"
            + (is3d ? "3D" : "2D") + " source;\n";
    source += std::string{"layout(std430, binding = 0) writeonly buffer Texels { "} + scalar + " texels[]; };\n";
    source += "void main() {\n";
    source += is3d ? "    ivec3 size = imageSize(source);\n" : "    ivec3 size = ivec3(imageSize(source), 1);\n";
    source += "    ivec3 p = ivec3(gl_GlobalInvocationID);\n";
    source += "    if (any(greaterThanEqual(p, size))) { return; }\n";
    source += std::string{"    "} + prefix + "vec4 v = imageLoad(source, " + (is3d ? "p" : "p.xy") + ");\n";
    source += "    int base = ((p.z * size.y + p.y) * size.x + p.x) * 4;\n";
    source += "    texels[base] = v.x;\n    texels[base + 1] = v.y;\n";
    source += "    texels[base + 2] = v.z;\n    texels[base + 3] = v.w;\n}\n";
    return source;
}

void encodeTexel(ImageFormat const &format, std::uint32_t const (&components)[4], std::byte *texel) {
    auto const bytes = format.bitsPerChannel / 8u;
    for (unsigned c = 0; c < format.channelCount; ++c) {
        auto stored = std::uint32_t{0};
        if (format.encoding == Encoding::Normalized) {
            auto value = 0.0f;
            std::memcpy(&value, &components[c], sizeof value);
            auto const maxValue = static_cast<float>((1u << format.bitsPerChannel) - 1u);
            auto const clamped = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
            stored = static_cast<std::uint32_t>(std::lround(clamped * maxValue));
        } else {
            stored = components[c];
        }
        // Little-endian truncation keeps the low `bytes` bytes, i.e. the
        // value modulo 2^bits, which is how integer stores behave.
        std::memcpy(texel + c * bytes, &stored, bytes);
    }
}

} // namespace tc::opengl
