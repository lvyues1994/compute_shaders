#pragma once

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

#include <tc/devices.hpp>

namespace examples {

struct CommandLine {
    tc::DeviceRequest device;
    int size = 0; // example-specific problem size; 0 keeps the default
};

inline void printUsage(char const *program) {
    std::cerr << "usage: " << program << " [--backend <name>] [--device <name filter>] [--size <n>]\n"
              << "backends:";
    for (auto const &name : tc::availableBackends()) {
        std::cerr << " " << name;
    }
    std::cerr << "\n";
}

// Parses `--backend vulkan --device llvmpipe --size 512`.
inline CommandLine parseCommandLine(int const argc, char **argv) {
    auto line = CommandLine{};
    for (int i = 1; i < argc; ++i) {
        auto const arg = std::string_view{argv[i]};
        auto const hasValue = i + 1 < argc;
        if (arg == "--backend" and hasValue) {
            line.device.backend = argv[++i];
        } else if (arg == "--device" and hasValue) {
            line.device.deviceNameContains = argv[++i];
        } else if (arg == "--size" and hasValue) {
            line.size = std::atoi(argv[++i]);
        } else {
            printUsage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    return line;
}

} // namespace examples
