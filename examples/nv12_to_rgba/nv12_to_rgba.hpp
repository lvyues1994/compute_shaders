#pragma once

#include <tc/tc.hpp>

namespace examples {

using namespace tc;

// Converts an NV12 frame (full-resolution Y plane, half-resolution
// interleaved UV plane) to RGBA. The planes are sampled textures, so the r8
// and rg8 formats work on OpenGL ES too; the colour matrix and offsets select
// the YUV standard (BT.601 / BT.709, limited or full range).
struct TC_KERNEL Nv12ToRgba {
    static constexpr char fileLocation[] = "nv12_to_rgba";
    uvec3 local_size{16, 16, 1};

    TextureBinding<gpu::R8, Dim::D2, cpu::R8, 0> luma;
    TextureBinding<gpu::RG8, Dim::D2, cpu::RG8, 1> chroma;
    ImageBinding<gpu::RGBA8, Dim::D2, cpu::RGBA8, 2> rgba;
    Uniform<mat3, 0> yuvToRgb{};
    Uniform<vec3, 1> yuvOffset{};

    void main() {
        ivec2 p = ivec2(gl_GlobalInvocationID.xy());
        ivec2 size = imageSize(rgba);
        if (p.x >= size.x or p.y >= size.y) {
            return;
        }
        float y = texelFetch(luma, p, 0).x;
        vec2 uv = texelFetch(chroma, p / 2, 0).xy();
        vec3 rgb = yuvToRgb.value * (vec3{y, uv.x, uv.y} - yuvOffset.value);
        imageStore(rgba, p, vec4(clamp(rgb, 0.0f, 1.0f), 1.0f));
    }
};

} // namespace examples
