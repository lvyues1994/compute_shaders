// tc_bench: times every benchmark case on every requested backend.
//
//   tc_bench [--backend cpu,vulkan] [--device llvmpipe] [--case transpose]
//            [--scale 0.25] [--budget 1000] [--iterations 50] [--csv out.csv]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <tc/devices.hpp>

#include <game_of_life/simulate.hpp>
#include <nv12_to_rgba/convert.hpp>

#include "harness.hpp"
#include "kernels.hpp"

namespace bench {
namespace {

// `base` scaled, rounded to a positive multiple of `multiple`.
std::size_t scaled(double const base, double const scale, std::size_t const multiple) {
    auto const blocks = std::llround(base * scale / static_cast<double>(multiple));
    return static_cast<std::size_t>(std::max<long long>(blocks, 1)) * multiple;
}

// Workgroups covering `n` invocations, at most 32768 per row (see linearIndex).
uvec3 groups1d(std::size_t const n, uint const local) {
    auto const total = static_cast<uint>((n + local - 1) / local);
    auto const x = std::min(total, 32768u);
    return uvec3{x, (total + x - 1) / x, 1u};
}

uvec3 groups2d(ivec2 const size, uint const local) {
    return uvec3{(static_cast<uint>(size.x) + local - 1) / local, (static_cast<uint>(size.y) + local - 1) / local, 1u};
}

template <class T>
void fillRamp(BufferResource<T> &buffer) {
    for (std::size_t i = 0; i < buffer.size(); ++i) {
        buffer[i] = T(static_cast<float>(i % 1024) * (1.0f / 1024.0f));
    }
}

struct CopyWorkload final : Workload {
    CopyWorkload(Device &device, std::size_t const n) : source(n), target(n), groups(groups1d(n, 256)) {
        fillRamp(source);
        kernel.source.attach(source);
        kernel.target.attach(target);
        device.uploadBuffer(source);
        device.uploadBuffer(target);
        device.useKernel(kernel);
        device.bindBuffer(kernel.source);
        device.bindBuffer(kernel.target);
        bytesPerRun = 2.0 * static_cast<double>(n * sizeof(vec4));
    }
    void run(Device &device) override { device.execute(kernel, groups); }

    BufferResource<vec4> source;
    BufferResource<vec4> target;
    CopyVec4 kernel;
    uvec3 groups;
};

struct SaxpyWorkload final : Workload {
    SaxpyWorkload(Device &device, std::size_t const n) : x(n), y(n), groups(groups1d(n, 256)) {
        fillRamp(x);
        fillRamp(y);
        kernel.x.attach(x);
        kernel.y.attach(y);
        kernel.a = 0.5f;
        device.uploadBuffer(x);
        device.uploadBuffer(y);
        device.useKernel(kernel);
        device.bindBuffer(kernel.x);
        device.bindBuffer(kernel.y);
        device.setUniform(kernel.a);
        bytesPerRun = 3.0 * static_cast<double>(n * sizeof(float));
    }
    void run(Device &device) override { device.execute(kernel, groups); }

    BufferResource<float> x;
    BufferResource<float> y;
    Saxpy kernel;
    uvec3 groups;
};

struct ReduceWorkload final : Workload {
    ReduceWorkload(Device &device, std::size_t const n)
        : values(n), groups(groups1d(n, 256)), sums(std::size_t{groups.x} * groups.y) {
        fillRamp(values);
        kernel.values.attach(values);
        kernel.sums.attach(sums);
        device.uploadBuffer(values);
        device.uploadBuffer(sums);
        device.useKernel(kernel);
        device.bindBuffer(kernel.values);
        device.bindBuffer(kernel.sums);
        bytesPerRun = static_cast<double>((values.size() + sums.size()) * sizeof(float));
    }
    void run(Device &device) override { device.execute(kernel, groups); }

    BufferResource<float> values;
    uvec3 groups;
    BufferResource<float> sums;
    ReduceShared kernel;
};

template <class Kernel>
struct TransposeWorkload final : Workload {
    TransposeWorkload(Device &device, int const side)
        : size{side, side}, input(std::size_t(side) * side), output(std::size_t(side) * side) {
        fillRamp(input);
        kernel.input.attach(input);
        kernel.output.attach(output);
        kernel.size = size;
        device.uploadBuffer(input);
        device.uploadBuffer(output);
        device.useKernel(kernel);
        device.bindBuffer(kernel.input);
        device.bindBuffer(kernel.output);
        device.setUniform(kernel.size);
        bytesPerRun = 2.0 * static_cast<double>(input.size() * sizeof(float));
    }
    void run(Device &device) override { device.execute(kernel, groups2d(size, 16)); }

    ivec2 size;
    BufferResource<float> input;
    BufferResource<float> output;
    Kernel kernel;
};

struct ParticlesStd430Workload final : Workload {
    ParticlesStd430Workload(Device &device, std::size_t const n) : particles(n), groups(groups1d(n, 256)) {
        for (std::size_t i = 0; i < n; ++i) {
            auto const f = static_cast<float>(i % 4096);
            particles[i] = Particle{vec4{f, 2.0f * f, 0.5f, 1.0f + f}, vec4{1.0f, 0.0f, -1.0f, 0.0f}};
        }
        kernel.particles.attach(particles);
        kernel.dt = 1.0f / 60.0f;
        device.uploadBuffer(particles);
        device.useKernel(kernel);
        device.bindBuffer(kernel.particles);
        device.setUniform(kernel.dt);
        bytesPerRun = 2.0 * static_cast<double>(n * sizeof(Particle));
    }
    void run(Device &device) override { device.execute(kernel, groups); }

    BufferResource<Particle> particles;
    ParticlesStd430 kernel;
    uvec3 groups;
};

template <class Kernel>
using TableOf = typename decltype(Kernel::particles)::Resource;

template <class Kernel>
struct ParticlesTableWorkload final : Workload {
    ParticlesTableWorkload(Device &device, std::size_t const n) : particles(n), groups(groups1d(n, 256)) {
        for (std::size_t i = 0; i < n; ++i) {
            auto const f = static_cast<float>(i % 4096);
            particles.set(i, ParticleData{vec3{f, 2.0f * f, 0.5f}, 1.0f + f, vec3{1.0f, 0.0f, -1.0f}, 0.0f});
        }
        kernel.particles.attach(particles);
        kernel.dt = 1.0f / 60.0f;
        device.uploadTable(particles);
        device.useKernel(kernel);
        device.bindTable(kernel.particles);
        device.setUniform(kernel.dt);
        bytesPerRun = 2.0 * static_cast<double>(n * TableOf<Kernel>::recordWords * sizeof(uint));
    }
    void run(Device &device) override { device.execute(kernel, groups); }

    TableOf<Kernel> particles;
    Kernel kernel;
    uvec3 groups;
};

// One instance in 16 is enabled.
template <class Kernel>
struct CullWorkload final : Workload {
    using Table = typename decltype(Kernel::instances)::Resource;

    CullWorkload(Device &device, std::size_t const n) : instances(n), visible(n), groups(groups1d(n, 256)) {
        for (std::size_t i = 0; i < n; ++i) {
            auto const f = static_cast<float>(i % 512);
            auto instance = Instance{};
            instance.rotation = mat3(1.0f);
            instance.translation = vec3{f, 1.0f, -f};
            instance.color = vec4{0.5f, f / 512.0f, 1.0f, 1.0f};
            instance.center = vec3{0.0f, 2.0f, 0.0f};
            instance.radius = 1.0f + f;
            instance.flags = i % 16 == 0 ? 1u : 0u;
            instances.set(i, instance);
        }
        kernel.instances.attach(instances);
        kernel.visible.attach(visible);
        device.uploadTable(instances);
        device.uploadBuffer(visible);
        device.useKernel(kernel);
        device.bindTable(kernel.instances);
        device.bindBuffer(kernel.visible);
        // Needed traffic: every flag, the other 20 words and one vec4 written
        // for each enabled instance.
        auto const enabled = static_cast<double>(n) / 16.0;
        bytesPerRun = static_cast<double>(n) * 4.0 + enabled * (20.0 * 4.0 + 16.0);
    }
    void run(Device &device) override { device.execute(kernel, groups); }

    Table instances;
    BufferResource<vec4> visible;
    uvec3 groups;
    Kernel kernel;
};

struct PointsVec4Workload final : Workload {
    PointsVec4Workload(Device &device, std::size_t const n) : points(n), groups(groups1d(n, 256)) {
        fillRamp(points);
        kernel.points.attach(points);
        device.uploadBuffer(points);
        device.useKernel(kernel);
        device.bindBuffer(kernel.points);
        bytesPerRun = 2.0 * static_cast<double>(n * sizeof(vec4));
    }
    void run(Device &device) override { device.execute(kernel, groups); }

    BufferResource<vec4> points;
    ScalePointsVec4 kernel;
    uvec3 groups;
};

struct PointsTableWorkload final : Workload {
    PointsTableWorkload(Device &device, std::size_t const n) : points(n), groups(groups1d(n, 256)) {
        for (std::size_t i = 0; i < n; ++i) {
            points.set(i, vec3{static_cast<float>(i % 1024) * (1.0f / 1024.0f)});
        }
        kernel.points.attach(points);
        device.uploadTable(points);
        device.useKernel(kernel);
        device.bindTable(kernel.points);
        bytesPerRun = 2.0 * static_cast<double>(n * 3 * sizeof(float));
    }
    void run(Device &device) override { device.execute(kernel, groups); }

    TableResource<vec3> points;
    ScalePointsTable kernel;
    uvec3 groups;
};

template <class Kernel>
struct ShadeWorkload final : Workload {
    using Table = typename decltype(Kernel::inputs)::Resource;
    using Element = typename Table::value_type;

    ShadeWorkload(Device &device, std::size_t const n) : inputs(n), shaded(n), groups(groups1d(n, 256)) {
        for (std::size_t i = 0; i < n; ++i) {
            auto const f = static_cast<float>(i % 512);
            auto element = Element{};
            element.transform = mat4{vec4{1.0f, 0.0f, 0.0f, 0.0f}, vec4{0.0f, 1.0f, 0.0f, 0.0f},
                                     vec4{0.0f, 0.0f, 1.0f, 0.0f}, vec4{f, 0.0f, -f, 1.0f}};
            element.color = vec4{0.5f, f / 512.0f, 0.25f, 1.0f};
            element.normal = normalize(vec3{f - 256.0f, 1.0f, 2.0f});
            element.uv = vec2{f / 512.0f, 0.5f};
            inputs.set(i, element);
        }
        kernel.inputs.attach(inputs);
        kernel.shaded.attach(shaded);
        device.uploadTable(inputs);
        device.uploadBuffer(shaded);
        device.useKernel(kernel);
        device.bindTable(kernel.inputs);
        device.bindBuffer(kernel.shaded);
        bytesPerRun = static_cast<double>(n) * (Table::recordWords * 4.0 + 16.0);
    }
    void run(Device &device) override { device.execute(kernel, groups); }

    Table inputs;
    BufferResource<vec4> shaded;
    uvec3 groups;
    Kernel kernel;
};

struct Nv12Workload final : Workload {
    Nv12Workload(Device &device, ivec2 const size)
        : frame(examples::makeTestFrame(size)), rgba(size), groups(groups2d(size, 16)) {
        auto const standard = examples::bt601Limited();
        kernel.luma.attach(frame.luma);
        kernel.chroma.attach(frame.chroma);
        kernel.rgba.attach(rgba);
        kernel.yuvToRgb = standard.toRgb;
        kernel.yuvOffset = standard.offset;
        device.uploadImage(kernel.luma);
        device.uploadImage(kernel.chroma);
        device.uploadImage(kernel.rgba);
        device.useKernel(kernel);
        device.bindTexture(kernel.luma);
        device.bindTexture(kernel.chroma);
        device.bindImage(kernel.rgba);
        device.setUniform(kernel.yuvToRgb);
        device.setUniform(kernel.yuvOffset);
        auto const pixels = static_cast<double>(size.x) * size.y;
        bytesPerRun = pixels * (1.0 + 0.5 + 4.0); // Y, quarter-size UV pairs, RGBA8 out
    }
    void run(Device &device) override { device.execute(kernel, groups); }

    examples::Nv12Frame frame;
    examples::RgbaImage rgba;
    examples::Nv12ToRgba kernel;
    uvec3 groups;
};

// Two generations per run, so the grid returns to the front image.
struct GameOfLifeWorkload final : Workload {
    GameOfLifeWorkload(Device &device, int const cells)
        : front(examples::randomGrid(cells, 7u)), back(front.extent()), groups(groups2d(ivec2{cells}, 16)) {
        kernel.pad = 1;
        kernel.inData.attach(front);
        kernel.outData.attach(back);
        device.uploadImage(kernel.inData);
        device.uploadImage(kernel.outData);
        device.useKernel(kernel);
        device.setUniform(kernel.pad);
        bytesPerRun = 2.0 * 2.0 * static_cast<double>(front.size() * sizeof(uint)); // r32ui read + write
    }
    void run(Device &device) override {
        step(device, front, back);
        step(device, back, front);
    }
    void step(Device &device, examples::Grid &from, examples::Grid &to) {
        kernel.inData.attach(from);
        kernel.outData.attach(to);
        device.bindImage(kernel.inData);
        device.bindImage(kernel.outData);
        device.execute(kernel, groups);
    }

    examples::Grid front;
    examples::Grid back;
    examples::GameOfLife kernel;
    uvec3 groups;
};

template <class W, class... Args>
std::unique_ptr<Workload> make(Args &&...args) {
    return std::make_unique<W>(std::forward<Args>(args)...);
}

int side(double const base, double const scale) {
    return static_cast<int>(scaled(base, std::sqrt(scale), 16));
}

std::vector<Case> allCases() {
    return {
        {"copy_vec4", [](Device &d, double s) { return make<CopyWorkload>(d, scaled(1 << 22, s, 256)); }},
        {"saxpy", [](Device &d, double s) { return make<SaxpyWorkload>(d, scaled(1 << 23, s, 256)); }},
        {"reduce_shared", [](Device &d, double s) { return make<ReduceWorkload>(d, scaled(1 << 22, s, 256)); }},
        {"transpose_naive",
         [](Device &d, double s) { return make<TransposeWorkload<TransposeNaive>>(d, side(2048, s)); }},
        {"transpose_tiled",
         [](Device &d, double s) { return make<TransposeWorkload<TransposeTiled>>(d, side(2048, s)); }},
        {"particles_std430",
         [](Device &d, double s) { return make<ParticlesStd430Workload>(d, scaled(1 << 21, s, 256)); }},
        {"particles_table",
         [](Device &d, double s) {
             return make<ParticlesTableWorkload<ParticlesTable>>(d, scaled(1 << 21, s, 256));
         }},
        {"particles_soa",
         [](Device &d, double s) { return make<ParticlesTableWorkload<ParticlesSoa>>(d, scaled(1 << 21, s, 256)); }},
        {"cull_aos", [](Device &d, double s) { return make<CullWorkload<CullAos>>(d, scaled(1 << 21, s, 256)); }},
        {"cull_soa", [](Device &d, double s) { return make<CullWorkload<CullSoa>>(d, scaled(1 << 21, s, 256)); }},
        {"shade_plain",
         [](Device &d, double s) { return make<ShadeWorkload<ShadePlain>>(d, scaled(1 << 21, s, 256)); }},
        {"shade_encoded",
         [](Device &d, double s) { return make<ShadeWorkload<ShadeEncoded>>(d, scaled(1 << 21, s, 256)); }},
        {"points_vec4_std430",
         [](Device &d, double s) { return make<PointsVec4Workload>(d, scaled(1 << 22, s, 256)); }},
        {"points_vec3_table",
         [](Device &d, double s) { return make<PointsTableWorkload>(d, scaled(1 << 22, s, 256)); }},
        {"nv12_to_rgba",
         [](Device &d, double s) {
             return make<Nv12Workload>(d, ivec2{side(1920, s), static_cast<int>(scaled(1080, std::sqrt(s), 16))});
         }},
        {"game_of_life", [](Device &d, double s) { return make<GameOfLifeWorkload>(d, side(2048, s)); }},
    };
}

struct Options {
    std::vector<std::string> backends;
    std::string deviceNameContains;
    std::vector<std::string> caseFilters;
    std::string csvPath;
    Settings settings;
    bool list = false;
};

std::vector<std::string> splitList(std::string_view const text) {
    auto items = std::vector<std::string>{};
    auto stream = std::istringstream{std::string{text}};
    for (auto item = std::string{}; std::getline(stream, item, ',');) {
        if (not item.empty()) {
            items.push_back(item);
        }
    }
    return items;
}

[[noreturn]] void usage(char const *program) {
    std::cerr << "usage: " << program
              << " [--backend a,b] [--device <name filter>] [--case <substring>,...] [--scale <factor>]\n"
                 "       [--budget <ms per case>] [--iterations <max>] [--csv <path>] [--list]\nbackends:";
    for (auto const &name : availableBackends()) {
        std::cerr << " " << name;
    }
    std::cerr << "\n";
    std::exit(EXIT_FAILURE);
}

Options parse(int const argc, char **argv) {
    auto options = Options{};
    for (int i = 1; i < argc; ++i) {
        auto const arg = std::string_view{argv[i]};
        auto const value = [&]() -> std::string_view {
            if (i + 1 >= argc) {
                usage(argv[0]);
            }
            return argv[++i];
        };
        if (arg == "--backend") {
            options.backends = splitList(value());
        } else if (arg == "--device") {
            options.deviceNameContains = value();
        } else if (arg == "--case") {
            options.caseFilters = splitList(value());
        } else if (arg == "--scale") {
            options.settings.scale = std::atof(std::string{value()}.c_str());
        } else if (arg == "--budget") {
            options.settings.budgetMilliseconds = std::atof(std::string{value()}.c_str());
        } else if (arg == "--iterations") {
            options.settings.maxIterations = std::max(1, std::atoi(std::string{value()}.c_str()));
            options.settings.minIterations = std::min(options.settings.minIterations, options.settings.maxIterations);
        } else if (arg == "--csv") {
            options.csvPath = value();
        } else if (arg == "--list") {
            options.list = true;
        } else {
            usage(argv[0]);
        }
    }
    if (options.backends.empty()) {
        options.backends = availableBackends();
    }
    if (options.settings.scale <= 0.0) {
        usage(argv[0]);
    }
    return options;
}

bool selected(Case const &benchmark, std::vector<std::string> const &filters) {
    return filters.empty() or std::any_of(filters.begin(), filters.end(), [&](std::string const &filter) {
               return benchmark.name.find(filter) != std::string::npos;
           });
}

std::string format(std::optional<double> const value, int const precision) {
    if (not value) {
        return "-";
    }
    char text[32];
    std::snprintf(text, sizeof text, "%.*f", precision, *value);
    return text;
}

void printRow(std::string const &name, std::string const &gpuMedian, std::string const &gpuMin,
              std::string const &wall, std::string const &bandwidth, std::string const &iterations) {
    std::printf("  %-18s %12s %12s %12s %10s %6s\n", name.c_str(), gpuMedian.c_str(), gpuMin.c_str(), wall.c_str(),
                bandwidth.c_str(), iterations.c_str());
}

int run(Options const &options) {
    auto const cases = allCases();
    if (options.list) {
        for (auto const &benchmark : cases) {
            std::cout << benchmark.name << "\n";
        }
        return EXIT_SUCCESS;
    }
    auto csv = std::ofstream{};
    if (not options.csvPath.empty()) {
        csv.open(options.csvPath);
        csv << "case,backend,device,iterations,gpu_median_ms,gpu_min_ms,wall_ms,bytes_per_run,gb_per_s\n";
    }
    for (auto const &backend : options.backends) {
        auto device = std::unique_ptr<Device>{};
        try {
            device = createDevice(DeviceRequest{backend, options.deviceNameContains});
        } catch (DeviceError const &error) {
            std::cout << backend << ": skipped (" << error.what() << ")\n\n";
            continue;
        }
        auto const info = device->info();
        std::cout << info.backend << ": " << info.deviceName << " (timers: " << (info.timers ? "yes" : "no")
                  << ", unified memory: " << (info.unifiedMemory ? "yes" : "no") << ")\n";
        printRow("case", "gpu ms", "gpu min", "wall ms", "GB/s", "runs");
        for (auto const &benchmark : cases) {
            if (not selected(benchmark, options.caseFilters)) {
                continue;
            }
            try {
                auto const m = measure(benchmark, *device, options.settings);
                printRow(benchmark.name, format(m.gpuMedian, 3), format(m.gpuMin, 3), format(m.wallMilliseconds, 3),
                         format(m.gigabytesPerSecond(), 1), std::to_string(m.iterations));
                if (csv.is_open()) {
                    csv << benchmark.name << "," << info.backend << ",\"" << info.deviceName << "\"," << m.iterations
                        << "," << format(m.gpuMedian, 4) << "," << format(m.gpuMin, 4) << ","
                        << format(m.wallMilliseconds, 4) << "," << m.bytesPerRun << ","
                        << format(m.gigabytesPerSecond(), 2) << "\n";
                }
            } catch (DeviceError const &error) {
                std::cout << "  " << benchmark.name << ": n/a (" << error.what() << ")\n";
            }
            std::cout.flush();
        }
        std::cout << "\n";
    }
    return EXIT_SUCCESS;
}

} // namespace
} // namespace bench

int main(int argc, char **argv) {
    return bench::run(bench::parse(argc, argv));
}
