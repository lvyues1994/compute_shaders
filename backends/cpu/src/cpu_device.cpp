#include <tc/cpu/cpu_device.hpp>

#include <algorithm>
#include <atomic>
#include <string>
#include <thread>
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
        return DeviceInfo{"cpu", "host (" + std::to_string(threadCount) + " threads)", true};
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

private:
    std::uint32_t threadCount;
};

void CpuDevice::doExecute(Dispatch const &dispatch) {
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
