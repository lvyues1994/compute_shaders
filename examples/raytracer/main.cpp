#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

#include <examples/cli.hpp>
#include <tc/devices.hpp>

#include "render.hpp"

namespace {

int run(examples::CommandLine const &line) {
    auto device = tc::createDevice(line.device);
    auto const info = device->info();
    std::cout << "backend " << info.backend << ": " << info.deviceName << "\n";

    auto const width = line.size > 0 ? line.size : 640;
    auto const size = tc::ivec2{width, width * 3 / 4};
    auto const start = std::chrono::steady_clock::now();
    auto const picture = examples::render(*device, size, 0.6f);
    auto const elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start);

    auto const path = "raytracer_" + info.backend + ".ppm";
    if (not examples::writePpm(picture, path)) {
        std::cerr << "cannot write " << path << "\n";
        return EXIT_FAILURE;
    }
    std::cout << size.x << "x" << size.y << " rendered in " << elapsed.count() << " ms -> " << path << "\n";
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
