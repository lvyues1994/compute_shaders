#pragma once

// The fixture of the equivalence tests: every test runs on the CPU backend
// and on one GPU backend, and compares the results. Test files add TEST_P
// cases for `Equivalence`; equivalence_tests.cpp instantiates them all.

#include <cctype>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <tc/cpu/cpu_device.hpp>
#include <tc/devices.hpp>

namespace tc::testing_support {

struct BackendCase {
    std::string backend;
    std::string deviceNameContains;
};

inline std::string caseName(testing::TestParamInfo<BackendCase> const &info) {
    auto name = info.param.backend + (info.param.deviceNameContains.empty() ? "" : "_" + info.param.deviceNameContains);
    for (auto &c : name) {
        c = std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
    }
    return name;
}

inline std::vector<BackendCase> gpuCases() {
    auto cases = std::vector<BackendCase>{};
    for (auto const &backend : availableBackends()) {
        if (backend == "cpu") {
            continue;
        }
        cases.push_back(BackendCase{backend, ""});
        cases.push_back(BackendCase{backend, "llvmpipe"});
    }
    return cases;
}

struct Equivalence : testing::TestWithParam<BackendCase> {
    void SetUp() override {
        auto request = DeviceRequest{};
        request.backend = GetParam().backend;
        request.deviceNameContains = GetParam().deviceNameContains;
        try {
            gpu = createDevice(request);
        } catch (DeviceError const &error) {
            GTEST_SKIP() << "backend unavailable: " << error.what();
        }
        cpu = cpu::createDevice();
    }

    // Runs `body` on both devices; skips when the GPU backend reports that
    // the kernel cannot run there (e.g. features OpenGL ES lacks).
    template <class Result>
    std::pair<Result, Result> runBoth(std::function<Result(Device &)> const &body) {
        auto expected = body(*cpu);
        try {
            return {expected, body(*gpu)};
        } catch (DeviceError const &error) {
            auto const message = std::string{error.what()};
            if (message.find("cannot run on OpenGL ES") != std::string::npos
                or message.find("which this device does not support") != std::string::npos) {
                throw SkipKernel{error.what()};
            }
            throw;
        }
    }

    struct SkipKernel {
        std::string reason;
    };

    std::unique_ptr<Device> gpu;
    std::unique_ptr<Device> cpu;
};

template <class T>
std::vector<T> contents(BufferResource<T> const &buffer) {
    return std::vector<T>(buffer.begin(), buffer.end());
}

} // namespace tc::testing_support
