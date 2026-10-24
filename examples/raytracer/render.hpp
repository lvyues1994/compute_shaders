#pragma once

#include <examples/ppm.hpp>
#include <tc/device.hpp>

#include "raytracer.hpp"

namespace examples {

using Picture = RgbaImage;

inline Picture render(tc::Device &device, tc::ivec2 const size, float const time) {
    auto picture = Picture{size};
    auto kernel = Raytracer{};
    kernel.time = time;
    kernel.image.attach(picture);
    device.uploadImage(kernel.image);
    device.useKernel(kernel);
    device.bindImage(kernel.image);
    device.setUniform(kernel.time);
    device.execute(kernel, tc::uvec3{(static_cast<tc::uint>(size.x) + 7u) / 8u,
                                     (static_cast<tc::uint>(size.y) + 7u) / 8u, 1u});
    device.downloadImage(kernel.image);
    return picture;
}

} // namespace examples
