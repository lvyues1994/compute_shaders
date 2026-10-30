#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <tc/device.hpp>

namespace bench {

// One problem prepared on one device: resources uploaded and bound, so run()
// only queues the dispatches of one iteration.
struct Workload {
    Workload() = default;
    Workload(Workload const &) = delete;
    Workload &operator=(Workload const &) = delete;
    virtual ~Workload() = default;

    virtual void run(tc::Device &device) = 0;

    double bytesPerRun = 0.0; // compulsory traffic: bytes read plus bytes written
};

struct Case {
    std::string name;
    // Builds the workload; `scale` multiplies the default problem size.
    std::function<std::unique_ptr<Workload>(tc::Device &, double scale)> prepare;
};

struct Settings {
    double scale = 1.0;
    double budgetMilliseconds = 1000.0; // per case and device
    int minIterations = 3;
    int maxIterations = 50;
};

struct Measurement {
    int iterations = 0;
    double bytesPerRun = 0.0;
    double wallMilliseconds = 0.0; // per run: host clock around the queued runs and finish()
    std::optional<double> gpuMedian; // per run, from timer queries
    std::optional<double> gpuMin;

    // From the GPU median when timers measured something (software drivers
    // may report zero), else from the wall clock.
    double gigabytesPerSecond() const {
        auto const milliseconds = gpuMedian.value_or(0.0) > 0.0 ? *gpuMedian : wallMilliseconds;
        return milliseconds > 0.0 ? bytesPerRun / (milliseconds * 1e6) : 0.0;
    }
};

inline double millisecondsSince(std::chrono::steady_clock::time_point const start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

inline double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    auto const middle = values.size() / 2;
    return values.size() % 2 == 1 ? values[middle] : 0.5 * (values[middle - 1] + values[middle]);
}

// Sums the timings of each run; runs touched by a disjoint event are dropped.
inline std::vector<double> perRunTimes(std::vector<tc::KernelTiming> const &timings, int const iterations) {
    auto runs = std::vector<double>{};
    if (iterations <= 0 or timings.size() % static_cast<std::size_t>(iterations) != 0) {
        return runs;
    }
    auto const dispatchesPerRun = timings.size() / static_cast<std::size_t>(iterations);
    for (std::size_t run = 0; run < static_cast<std::size_t>(iterations); ++run) {
        auto total = 0.0;
        auto valid = true;
        for (std::size_t d = 0; d < dispatchesPerRun; ++d) {
            auto const &timing = timings[run * dispatchesPerRun + d];
            valid = valid and timing.milliseconds.has_value();
            total += timing.milliseconds.value_or(0.0);
        }
        if (valid) {
            runs.push_back(total);
        }
    }
    return runs;
}

// Warms up (pipeline creation, first-touch faults), sizes the iteration count
// to the time budget, then times the runs back to back.
inline Measurement measure(Case const &benchmark, tc::Device &device, Settings const &settings) {
    auto workload = benchmark.prepare(device, settings.scale);
    workload->run(device);
    device.finish();
    auto const probeStart = std::chrono::steady_clock::now();
    workload->run(device);
    device.finish();
    auto const once = std::max(millisecondsSince(probeStart), 1e-3);

    auto result = Measurement{};
    result.bytesPerRun = workload->bytesPerRun;
    result.iterations = std::clamp(static_cast<int>(settings.budgetMilliseconds / once), settings.minIterations,
                                   settings.maxIterations);
    auto const timers = device.info().timers;
    device.enableTimers(timers);
    auto const start = std::chrono::steady_clock::now();
    for (int i = 0; i < result.iterations; ++i) {
        workload->run(device);
    }
    device.finish();
    result.wallMilliseconds = millisecondsSince(start) / result.iterations;
    if (timers) {
        auto const runs = perRunTimes(device.takeTimings(), result.iterations);
        device.enableTimers(false);
        if (not runs.empty()) {
            result.gpuMedian = median(runs);
            result.gpuMin = *std::min_element(runs.begin(), runs.end());
        }
    }
    return result;
}

} // namespace bench
