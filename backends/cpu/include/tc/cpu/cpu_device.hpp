#pragma once

#include <cstdint>
#include <memory>

#include <tc/device.hpp>

namespace tc::cpu {

struct CpuDeviceConfig {
    std::uint32_t threadCount = 0; // 0 selects std::thread::hardware_concurrency()
};

// Runs kernels as plain C++: workgroups spread over worker threads, the
// invocations of one workgroup run sequentially. Kernels can be debugged
// with a normal debugger and sanitizers.
std::unique_ptr<Device> createDevice(CpuDeviceConfig const &config = {});

} // namespace tc::cpu
