#pragma once

#include <cstdint>
#include <utility>

#include <tc/device.hpp>

#include "game_of_life.hpp"

namespace examples {

using Grid = tc::BufferResource<tc::cpu::R8UI, tc::Dim::D2>;

// A grid of `cells` x `cells` live/dead cells inside a dead one-cell border,
// seeded from a fixed linear congruential sequence.
inline Grid randomGrid(int const cells, std::uint32_t seed) {
    auto grid = Grid{tc::ivec2{cells + 2, cells + 2}};
    for (int y = 1; y <= cells; ++y) {
        for (int x = 1; x <= cells; ++x) {
            seed = seed * 1664525u + 1013904223u;
            grid[tc::ivec2{x, y}].channels[0] = static_cast<std::uint8_t>((seed >> 28) < 5 ? 1 : 0);
        }
    }
    return grid;
}

// Runs `generations` steps on `device`, ping-ponging between two images that
// stay on the device; only the final grid is downloaded.
inline Grid simulate(tc::Device &device, Grid const &initial, int const generations) {
    auto front = initial;
    auto back = Grid{initial.extent()};
    auto kernel = GameOfLife{};
    kernel.pad = 1;
    kernel.inData.attach(front);
    kernel.outData.attach(back);
    device.uploadImage(kernel.inData);
    device.uploadImage(kernel.outData);

    auto const cells = initial.extent() - tc::ivec2{2 * kernel.pad};
    auto const groups = tc::uvec3{(static_cast<tc::uint>(cells.x) + 15u) / 16u,
                                  (static_cast<tc::uint>(cells.y) + 15u) / 16u, 1u};
    device.useKernel(kernel);
    device.setUniform(kernel.pad);
    auto *current = &front;
    auto *next = &back;
    for (int generation = 0; generation < generations; ++generation) {
        kernel.inData.attach(*current);
        kernel.outData.attach(*next);
        device.bindImage(kernel.inData);
        device.bindImage(kernel.outData);
        device.execute(kernel, groups);
        std::swap(current, next);
    }
    kernel.outData.attach(*current);
    device.downloadImage(kernel.outData);
    return *current;
}

inline int population(Grid const &grid) {
    auto alive = 0;
    for (auto const &cell : grid) {
        alive += cell.channels[0];
    }
    return alive;
}

} // namespace examples
