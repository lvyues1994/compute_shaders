#include <tc/cpu/cpu_device.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace tc::cpu {

namespace {

struct CpuDevice final : Device {
    explicit CpuDevice(CpuDeviceConfig const &config)
        : executor(config.executor != nullptr ? config.executor : makeThreadPool(config.threadCount)),
          subgroupSize(config.subgroupSize) {
        if (subgroupSize == 0 or subgroupSize > detail::kMaxSubgroupSize or (subgroupSize & (subgroupSize - 1)) != 0) {
            throw DeviceError{"CpuDeviceConfig::subgroupSize must be a power of two from 1 to 128"};
        }
    }

    DeviceInfo info() const override {
        auto result = DeviceInfo{};
        result.backend = "cpu";
        result.deviceName = "host (" + std::to_string(executor->concurrency()) + " threads)";
        result.unifiedMemory = true;
        result.timers = true;
        auto constexpr unlimited = std::numeric_limits<std::uint32_t>::max();
        auto &limits = result.limits;
        limits.storageBuffers = unlimited;
        limits.storageImages = unlimited;
        limits.sampledTextures = unlimited;
        limits.storageBufferBytes = std::numeric_limits<std::uint64_t>::max();
        limits.sharedBytes = unlimited;
        limits.workGroupInvocations = unlimited;
        limits.workGroupSize = uvec3{unlimited};
        limits.workGroupCount = uvec3{unlimited};
        limits.storageBufferOffsetAlignment = 1;
        limits.features = (KernelFeature::Double | KernelFeature::Subgroups)
                        | (KernelFeature::Int64 | KernelFeature::Int64Atomics);
        limits.subgroupSize = subgroupSize;
        limits.compressedFormats = (1u << kCompressedFormatCount) - 1u;
        limits.compressedArrayFormats = limits.compressedFormats;
        limits.imageDimension2D = unlimited;
        limits.imageDimension3D = unlimited;
        limits.imageArrayLayers = unlimited;
        return result;
    }

protected:
    // Host memory is the working copy, so transfers, bindings and uniforms
    // need no work: kernels read their members directly.
    std::optional<ExternalMemory> doAllocateBuffer(ResourceIdentity const &, std::size_t) override {
        return std::nullopt;
    }
    void doUploadBuffer(ResourceIdentity const &, ConstByteSpan, ByteRange) override {}
    void doDownloadBuffer(ResourceIdentity const &, ByteSpan, ByteRange) override {}
    void doUploadImage(ImageTransfer const &) override {}
    // Kernels sample the decoded copy.
    void doUploadCompressed(CompressedTransfer const &transfer) override { transfer.image->decode(); }
    void doDownloadImage(ImageTransfer const &) override {}
    // Device images live on the host here; execute() has already finished
    // the work before a copy.
    bool doCreateImage(ImageCreation const &) override { return true; }
    void doCopyBufferToImage(BufferImageCopy const &copy) override {
        copy.writeHost(ConstByteSpan{copy.host.data() + copy.offset, copy.bytes});
    }
    void doUseKernel(KernelUse const &) override { writtenFrames.clear(); }
    void doBindBuffer(BufferBindRequest const &) override {}
    void doBindImage(ImageBindRequest const &) override {}
    void doBindTexture(TextureBindRequest const &) override {}
    void doBindExternalTexture(ExternalTextureBindRequest const &request) override { request.image->snapshot(); }
    // Kernels read and write the CPU copy; every execute() stores it back.
    void doBindExternalImage(ExternalImageBindRequest const &request) override {
        request.image->snapshot();
        writtenFrames.push_back(request.image);
    }
    void doSetUniform(UniformWrite const &) override {}

    void doExecute(Dispatch const &dispatch) override;
    Submission doSubmit() override { return Submission{}; } // execute() already ran the work
    void doFinish() override {}
    std::vector<KernelTiming> doTakeTimings(std::uint64_t const last) override { return takeThrough(timings, last); }

private:
    void run(Dispatch const &dispatch, uvec3 groups) const;

    std::shared_ptr<ParallelExecutor> executor;
    std::uint32_t subgroupSize;
    std::vector<ExternalImage *> writtenFrames;
    std::vector<KernelTiming> timings;
};

void CpuDevice::doExecute(Dispatch const &dispatch) {
    // Earlier dispatches have completed, so indirect arguments are final.
    auto const groups = dispatch.indirect
                          ? uvec3{dispatch.indirect->host->x, dispatch.indirect->host->y, dispatch.indirect->host->z}
                          : dispatch.workGroups;
    auto const start = std::chrono::steady_clock::now();
    run(dispatch, groups);
    for (auto *frame : writtenFrames) {
        frame->writeBack();
    }
    if (not timersEnabled()) {
        return;
    }
    auto const elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start);
    timings.push_back(KernelTiming{std::string{dispatch.programName}, groups, elapsed.count(),
                                   dispatch.indirect.has_value(), nextSubmission()});
}

void CpuDevice::run(Dispatch const &dispatch, uvec3 const groups) const {
    auto const total = std::uint64_t{groups.x} * groups.y * groups.z;
    if (total == 0) {
        return;
    }
    // One task per thread, each with its own copy of the kernel, taking
    // workgroups until none are left.
    auto next = std::atomic<std::uint64_t>{0};
    auto const work = [this, &dispatch, &next, groups, total](std::uint32_t) {
        detail::subgroupSizeOnThisThread() = subgroupSize;
        auto const runWorkGroup = dispatch.makeWorker(groups);
        for (auto i = next.fetch_add(1); i < total; i = next.fetch_add(1)) {
            auto const x = static_cast<uint>(i % groups.x);
            auto const y = static_cast<uint>(i / groups.x % groups.y);
            auto const z = static_cast<uint>(i / (std::uint64_t{groups.x} * groups.y));
            runWorkGroup(uvec3{x, y, z});
        }
    };
    auto const tasks = static_cast<std::uint32_t>(std::min<std::uint64_t>(executor->concurrency(), total));
    executor->bulk(std::max(tasks, 1u), work);
}

} // namespace

std::unique_ptr<Device> createDevice(CpuDeviceConfig const &config) {
    return std::make_unique<CpuDevice>(config);
}

} // namespace tc::cpu
