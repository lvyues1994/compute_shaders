#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <examples/cli.hpp>
#include <tc/devices.hpp>

#include "convert.hpp"

namespace {

// Mean absolute difference (0-255 scale) between the decoded frame and the
// RGB pattern it was encoded from; chroma subsampling blurs bar edges.
double meanError(examples::RgbaImage const &image) {
    auto const size = image.extent();
    auto total = 0.0;
    for (int y = 0; y < size.y; ++y) {
        for (int x = 0; x < size.x; ++x) {
            auto const expected = examples::testPattern(tc::ivec2{x, y}, size) * 255.0f;
            auto const &pixel = image[tc::ivec2{x, y}];
            for (int c = 0; c < 3; ++c) {
                total += std::abs(static_cast<double>(pixel.channels[c]) - static_cast<double>(expected[c]));
            }
        }
    }
    return total / (3.0 * size.x * size.y);
}

int run(examples::CommandLine const &line) {
    auto device = tc::createDevice(line.device);
    auto const info = device->info();
    std::cout << "backend " << info.backend << ": " << info.deviceName << "\n";

    auto const width = line.size > 0 ? line.size : 1280;
    auto const size = tc::ivec2{width, width * 9 / 16} / 2 * 2;
    auto frame = examples::makeTestFrame(size);
    auto const start = std::chrono::steady_clock::now();
    auto const rgba = examples::convertNv12(*device, frame, examples::bt601Limited());
    auto const elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start);

    auto const path = "nv12_to_rgba_" + info.backend + ".ppm";
    examples::writePpm(rgba, path);
    std::cout << size.x << "x" << size.y << " NV12 -> RGBA in " << elapsed.count() << " ms, mean error "
              << meanError(rgba) << " / 255 -> " << path << "\n";
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
