#pragma once

#include <tc/tc.hpp>

namespace examples {

using namespace tc;

// Converts a platform frame (an Android AHardwareBuffer: camera, decoder,
// ImageReader) to packed RGBA8 without copying it: the frame is sampled in
// place and YUV is converted to RGB by the driver's sampler.
struct TC_KERNEL ExternalFrameToRgba {
    static constexpr char fileLocation[] = "external_frame_to_rgba";
    uvec3 local_size{16, 16, 1};

    ExternalTextureBinding<0> frame;
    BufferBinding<uint, 1> rgba;

    void main() {
        ivec2 p = ivec2(gl_GlobalInvocationID.xy());
        ivec2 size = textureSize(frame, 0);
        if (p.x >= size.x or p.y >= size.y) {
            return;
        }
        vec2 uv = (vec2(p) + 0.5f) / vec2(size);
        vec4 color = texture(frame, uv);
        rgba[p.y * size.x + p.x] = packUnorm4x8(clamp(color, 0.0f, 1.0f));
    }
};

} // namespace examples
