#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <examples/cli.hpp>
#include <tc/devices.hpp>

#include "float_adder.hpp"

namespace {

int run(examples::CommandLine const &line) {
    auto device = tc::createDevice(line.device);
    auto const info = device->info();
    std::cout << "backend " << info.backend << ": " << info.deviceName << "\n";

    auto const n = line.size > 0 ? line.size : 1 << 20;
    auto a = tc::BufferResource<float>{n};
    auto b = tc::BufferResource<float>{n};
    auto c = tc::BufferResource<float>{n};
    for (int i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = 0.5f * static_cast<float>(i);
    }

    auto kernel = examples::FloatAdder{};
    kernel.A.attach(a);
    kernel.B.attach(b);
    kernel.C.attach(c);

    auto const start = std::chrono::steady_clock::now();
    device->uploadBuffer(a);
    device->uploadBuffer(b);
    device->uploadBuffer(c);
    device->useKernel(kernel);
    device->bindBuffer(kernel.A);
    device->bindBuffer(kernel.B);
    device->bindBuffer(kernel.C);
    device->execute(kernel, tc::uvec3{(static_cast<tc::uint>(n) + 255u) / 256u, 1u, 1u});
    device->downloadBuffer(c);
    auto const elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start);

    auto mismatches = 0;
    for (int i = 0; i < n; ++i) {
        if (c[i] != a[i] + b[i]) {
            ++mismatches;
        }
    }
    std::cout << n << " additions in " << elapsed.count() << " ms, " << mismatches << " mismatches\n";
    return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
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
