#include <tc/cpu/cpu_device.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace tc::cpu {

namespace {

std::uint32_t resolveThreadCount(std::uint32_t const requested) {
    if (requested != 0) {
        return requested;
    }
    return std::max(1u, std::thread::hardware_concurrency());
}

struct CpuDevice final : Device {
    explicit CpuDevice(CpuDeviceConfig const &config) : threadCount(resolveThreadCount(config.threadCount)) {}

    DeviceInfo info() const override {
        auto result = DeviceInfo{};
        result.backend = "cpu";
        result.deviceName = "host (" + std::to_string(threadCount) + " threads)";
        result.unifiedMemory = true;
        result.timers = true;
        return result;
    }

protected:
    // Host memory is the working copy, so transfers, bindings and uniforms
    // need no work: kernels read their members directly.
    std::optional<ExternalMemory> doAllocateBuffer(ResourceIdentity const &, std::size_t) override {
        return std::nullopt;
    }
    void doUploadBuffer(ResourceIdentity const &, ConstByteSpan) override {}
    void doDownloadBuffer(ResourceIdentity const &, ByteSpan) override {}
    void doUploadImage(ImageTransfer const &) override {}
    void doDownloadImage(ImageTransfer const &) override {}
    void doUseKernel(KernelUse const &) override {}
    void doBindBuffer(BufferBindRequest const &) override {}
    void doBindImage(ImageBindRequest const &) override {}
    void doBindTexture(TextureBindRequest const &) override {}
    void doBindExternalTexture(ExternalTextureBindRequest const &request) override { request.image->snapshot(); }
    void doSetUniform(UniformWrite const &) override {}

    void doExecute(Dispatch const &dispatch) override;
    void doFinish() override {}
    std::vector<KernelTiming> doTakeTimings() override { return std::exchange(timings, {}); }

private:
    void run(Dispatch const &dispatch) const;

    std::uint32_t threadCount;
    std::vector<KernelTiming> timings;
};

void CpuDevice::doExecute(Dispatch const &dispatch) {
    if (not timersEnabled()) {
        run(dispatch);
        return;
    }
    auto const start = std::chrono::steady_clock::now();
    run(dispatch);
    auto const elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start);
    timings.push_back(KernelTiming{std::string{dispatch.programName}, dispatch.workGroups, elapsed.count()});
}

void CpuDevice::run(Dispatch const &dispatch) const {
    auto const groups = dispatch.workGroups;
    auto const total = std::uint64_t{groups.x} * groups.y * groups.z;
    if (total == 0) {
        return;
    }
    auto next = std::atomic<std::uint64_t>{0};
    auto const work = [&dispatch, &next, groups, total] {
        auto const runWorkGroup = dispatch.makeWorker();
        for (auto i = next.fetch_add(1); i < total; i = next.fetch_add(1)) {
            auto const x = static_cast<uint>(i % groups.x);
            auto const y = static_cast<uint>(i / groups.x % groups.y);
            auto const z = static_cast<uint>(i / (std::uint64_t{groups.x} * groups.y));
            runWorkGroup(uvec3{x, y, z});
        }
    };
    auto const helperCount = static_cast<std::size_t>(std::min<std::uint64_t>(threadCount, total) - 1);
    auto helpers = std::vector<std::thread>{};
    helpers.reserve(helperCount);
    for (std::size_t i = 0; i < helperCount; ++i) {
        helpers.emplace_back(work);
    }
    work();
    for (auto &helper : helpers) {
        helper.join();
    }
}

} // namespace

std::unique_ptr<Device> createDevice(CpuDeviceConfig const &config) {
    return std::make_unique<CpuDevice>(config);
}

} // namespace tc::cpu
