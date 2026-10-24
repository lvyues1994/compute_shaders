#include <chrono>
#include <cstdlib>
#include <iostream>

#include <examples/cli.hpp>
#include <tc/devices.hpp>

#include "simulate.hpp"

namespace {

void printGrid(examples::Grid const &grid) {
    auto const extent = grid.extent();
    for (int y = 1; y < extent.y - 1; ++y) {
        for (int x = 1; x < extent.x - 1; ++x) {
            std::cout << (grid[tc::ivec2{x, y}].channels[0] != 0 ? '#' : '.');
        }
        std::cout << '\n';
    }
}

int run(examples::CommandLine const &line) {
    auto device = tc::createDevice(line.device);
    auto const info = device->info();
    std::cout << "backend " << info.backend << ": " << info.deviceName << "\n";

    auto const cells = line.size > 0 ? line.size : 256;
    auto const generations = 100;
    auto const initial = examples::randomGrid(cells, 2025u);
    auto const start = std::chrono::steady_clock::now();
    auto const final = examples::simulate(*device, initial, generations);
    auto const elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start);

    std::cout << cells << "x" << cells << " grid, " << generations << " generations in " << elapsed.count()
              << " ms: population " << examples::population(initial) << " -> " << examples::population(final)
              << "\n";
    if (cells <= 48) {
        printGrid(final);
    }
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
    auto const line = examples::parseCommandLine(argc, argv);
    try {
        return run(line);
    } catch (tc::DeviceError const &error) {
        std::cerr << "device error: " << error.what() << "\n";
        return 2;
    }
}
