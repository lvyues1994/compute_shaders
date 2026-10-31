#include <tc/cpu/cpu_device.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tc/detail/aliasing.hpp>

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
        result.placedImages = true;
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
                        | (KernelFeature::Int64 | KernelFeature::Int64Atomics)
                        | static_cast<std::uint32_t>(KernelFeature::RayQuery);
        limits.accelerationStructures = unlimited;
        limits.maxInstances = 1u << 24; // the custom index has 24 bits
        limits.maxTriangles = unlimited;
        limits.maxGeometries = unlimited;
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
    // Device images, device buffers and heaps live on the host here;
    // execute() has already finished the work before a copy.
    bool doCreateImage(ImageCreation const &request) override {
        place(*request.identity, request.placement, request.hostBytes);
        return true;
    }
    bool doCreateHeap(HeapCreation const &) override { return true; }
    bool doCreateDeviceBuffer(BufferCreation const &request) override {
        place(*request.identity, request.placement, request.bytes);
        return true;
    }
    PlacementSize doBufferPlacement(std::size_t const bytes) override { return PlacementSize{bytes, 16}; }
    PlacementSize doImagePlacement(ImageCreation const &request) override { return PlacementSize{request.hostBytes, 16}; }
    void doCopyBufferToImage(BufferImageCopy const &copy) override {
        use(copy.buffer->key());
        use(copy.image->key());
        copy.writeHost(ConstByteSpan{copy.host.data() + copy.offset, copy.bytes});
    }
    void doCopyBuffer(BufferCopy const &copy) override {
        use(copy.source->key());
        use(copy.target->key());
        std::memmove(copy.targetHost.data() + copy.targetOffset, copy.sourceHost.data() + copy.sourceOffset, copy.bytes);
    }
    // Acceleration structures are BVHs on the host, built at once: the
    // work before has run already.
    void doCreateBottomLevel(BottomLevelBuild const &request) override {
        auto geometries = std::vector<detail::HostGeometry>{};
        for (auto const &geometry : *request.geometries) {
            auto host = detail::HostGeometry{};
            host.positions = ConstByteSpan{geometry.positionHost.data() + geometry.positionOffset,
                                           geometry.positionHost.size() - geometry.positionOffset};
            host.stride = geometry.stride;
            host.vertexCount = geometry.vertexCount;
            host.indices = ConstByteSpan{geometry.indexHost.data() + geometry.indexOffset,
                                         geometry.indexHost.size() - geometry.indexOffset};
            host.triangleCount = geometry.triangleCount;
            host.opaque = geometry.opaque;
            geometries.push_back(host);
        }
        auto blas = detail::buildBottomLevel(geometries);
        request.state->reference = ++references;
        bottomLevels[request.state->reference] = blas;
        request.state->host = std::move(blas);
    }
    void doCreateTopLevel(TopLevelCreation const &request) override {
        request.state->host = std::make_shared<detail::TopLevelHost const>();
    }
    void doBuildTopLevel(TopLevelBuild const &request) override {
        auto instances = std::vector<detail::HostInstance>{};
        for (std::uint32_t i = 0; i < request.count; ++i) {
            auto instance = TopLevelInstance{};
            std::memcpy(&instance, request.host.data() + request.offset + i * sizeof instance, sizeof instance);
            auto const reference = std::uint64_t{instance.reference.x} | std::uint64_t{instance.reference.y} << 32;
            auto const found = bottomLevels.find(reference);
            auto blas = found != bottomLevels.end() ? found->second.lock() : nullptr;
            if (blas == nullptr) {
                throw DeviceError{"buildTopLevel: instance " + std::to_string(i)
                                  + " refers to a bottom level that does not exist (any more)"};
            }
            auto host = detail::HostInstance{};
            host.objectToWorld = instance.transform;
            host.customIndex = instance.customIndexAndMask & 0xFFFFFFu;
            host.mask = instance.customIndexAndMask >> 24;
            host.flags = instance.offsetAndFlags >> 24;
            host.index = i;
            host.blas = std::move(blas);
            instances.push_back(std::move(host));
        }
        request.state->host = detail::buildTopLevel(std::move(instances));
    }
    void doBindAccelerationStructure(AccelerationBindRequest const &) override {}
    void doUseKernel(KernelUse const &) override { writtenFrames.clear(); }
    void doBindBuffer(BufferBindRequest const &request) override { use(request.identity->key()); }
    void doBindImage(ImageBindRequest const &request) override { use(request.identity->key()); }
    void doBindTexture(TextureBindRequest const &request) override { use(request.identity->key()); }
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

    void place(ResourceIdentity const &identity, std::optional<HeapRange> const &placement, std::size_t const bytes) {
        if (placement) {
            aliases.place(identity.key(), identity.watch(),
                          detail::AliasTracker::Range{placement->heap->key(), placement->offset, bytes});
            hostBytes[identity.key()] = placement->host;
        }
    }

    // A placed resource that an alias used since its own last use holds
    // garbage, as it may on a GPU: reading what another resource left
    // there then fails here every time.
    void use(void const *const key) {
        if (aliases.claim(key)) {
            std::memset(hostBytes.at(key), 0xCD, aliases.rangeOf(key)->bytes);
        }
    }

    std::shared_ptr<ParallelExecutor> executor;
    std::uint32_t subgroupSize;
    std::vector<ExternalImage *> writtenFrames;
    std::vector<KernelTiming> timings;
    detail::AliasTracker aliases;
    std::unordered_map<void const *, std::byte *> hostBytes; // of placed resources
    std::uint64_t references = 0; // the reference of the latest bottom level
    std::unordered_map<std::uint64_t, std::weak_ptr<detail::BottomLevelHost const>> bottomLevels;
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
