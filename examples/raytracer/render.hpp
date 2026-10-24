#pragma once

#include <fstream>
#include <string>

#include <tc/device.hpp>

#include "raytracer.hpp"

namespace examples {

using Picture = tc::BufferResource<tc::cpu::RGBA8, tc::Dim::D2>;

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

inline bool writePpm(Picture const &picture, std::string const &path) {
    auto file = std::ofstream{path, std::ios::binary};
    auto const size = picture.extent();
    file << "P6\n" << size.x << " " << size.y << "\n255\n";
    for (auto const &pixel : picture) {
        file.put(static_cast<char>(pixel.channels[0]));
        file.put(static_cast<char>(pixel.channels[1]));
        file.put(static_cast<char>(pixel.channels[2]));
    }
    return static_cast<bool>(file);
}

} // namespace examples
