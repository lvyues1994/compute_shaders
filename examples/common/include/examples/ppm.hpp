#pragma once

#include <fstream>
#include <string>

#include <tc/buffer.hpp>
#include <tc/pixel.hpp>

namespace examples {

using RgbaImage = tc::BufferResource<tc::cpu::RGBA8, tc::Dim::D2>;

// Binary PPM (P6) with the alpha channel dropped.
inline bool writePpm(RgbaImage const &image, std::string const &path) {
    auto file = std::ofstream{path, std::ios::binary};
    auto const size = image.extent();
    file << "P6\n" << size.x << " " << size.y << "\n255\n";
    for (auto const &pixel : image) {
        file.put(static_cast<char>(pixel.channels[0]));
        file.put(static_cast<char>(pixel.channels[1]));
        file.put(static_cast<char>(pixel.channels[2]));
    }
    return static_cast<bool>(file);
}

} // namespace examples
