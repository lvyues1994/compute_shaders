#pragma once

#include <array>
#include <cstdint>

#include <tc/tc.hpp>

namespace examples {

using namespace tc;

// Case 3 of the talk: one Game of Life generation. Cells live in a host
// image of 8-bit pixels (cpu::R8UI); on the GPU they are stored as r32ui, the
// integer format every OpenGL ES 3.1 and Vulkan device supports for storage
// images. The border of width `pad` is never written.
struct TC_KERNEL GameOfLife {
    static constexpr char fileLocation[] = "game_of_life";
    uvec3 local_size{16, 16, 1};

    ImageBinding<gpu::R32UI, Dim::D2, cpu::R8UI, 0> inData;
    ImageBinding<gpu::R32UI, Dim::D2, cpu::R8UI, 1> outData;
    Uniform<std::int32_t, 1> pad{1};

    static constexpr std::array<ivec2, 8> kIndices{
        ivec2{-1, -1}, ivec2{0, -1}, ivec2{1, -1},
        ivec2{-1, 0},                ivec2{1, 0},
        ivec2{-1, 1},  ivec2{0, 1},  ivec2{1, 1},
    };

    void main() {
        uvec2 gId = gl_GlobalInvocationID.xy();
        ivec2 coordinate = ivec2(gId.x + pad, gId.y + pad);
        ivec2 size = imageSize(inData);
        if (coordinate.x >= size.x - pad or coordinate.y >= size.y - pad) {
            return;
        }
        uint n = 0;
        bool alive = imageLoad(inData, coordinate).x;
        for (ivec2 const &offset : kIndices) {
            n += imageLoad(inData, coordinate + offset).x;
        }
        bool newState = (n == 3) or (alive and n == 2);
        imageStore(outData, coordinate, uvec4(newState));
    }
};

} // namespace examples
