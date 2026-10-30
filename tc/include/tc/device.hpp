#pragma once

#include <cstring>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <tc/buffer.hpp>
#include <tc/builtins.hpp>
#include <tc/external_image.hpp>
#include <tc/image.hpp>
#include <tc/kernel.hpp>
#include <tc/mat.hpp>
#include <tc/program.hpp>
#include <tc/span.hpp>
#include <tc/table.hpp>
#include <tc/texture.hpp>
#include <tc/uniform.hpp>
#include <tc/workgroup.hpp>

namespace tc {

// Misuse of a device (unbound resource, unknown kernel) or a failing graphics
// API call. Device creation failures are reported the same way.
struct DeviceError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct DeviceInfo {
    std::string backend;
    std::string deviceName;
    // createBuffer() memory is GPU-local as well as CPU-visible (mobile and
    // integrated GPUs, desktop GPUs with resizable BAR).
    bool unifiedMemory = false;
    // enableTimers() is available: GPU timestamp queries, or host time on
    // the CPU backend.
    bool timers = false;
};

// Device time of one execute() call.
struct KernelTiming {
    std::string program;
    uvec3 workGroups{};
    // nullopt when the GPU reported a disjoint event (a clock or power state
    // change) that invalidates the measurement.
    std::optional<double> milliseconds;
};

// Type-erased requests passed from the template API to a backend.
struct ImageTransfer {
    ResourceIdentity const *identity = nullptr;
    ImageFormat format{};
    Dim dim = Dim::D2;
    ivec3 extent{};
    std::function<void(ByteSpan)> writeTexels;     // upload: fill GPU-format texels
    std::function<void(ConstByteSpan)> readTexels; // download: consume GPU-format texels
};

struct BufferBindRequest {
    std::uint32_t binding = 0;
    ResourceIdentity const *identity = nullptr;
};

struct ImageBindRequest {
    std::uint32_t binding = 0;
    ResourceIdentity const *identity = nullptr;
    ImageFormat format{};
};

struct TextureBindRequest {
    std::uint32_t binding = 0;
    ResourceIdentity const *identity = nullptr;
    ImageFormat format{};
    Filter filter = Filter::Nearest;
    Wrap wrap = Wrap::ClampToEdge;
};

struct ExternalTextureBindRequest {
    std::uint32_t binding = 0;
    ExternalImage *image = nullptr;
};

// A scalar, vector or matrix value. `bytes` holds `columns` columns of
// `components` scalars each, `columnStride` bytes apart (16 for mat3).
struct UniformWrite {
    std::uint32_t location = 0;
    ScalarKind scalar = ScalarKind::Float;
    std::uint32_t components = 1;
    std::uint32_t columns = 1;
    std::uint32_t columnStride = 0;
    ConstByteSpan bytes;
};

struct KernelUse {
    std::string_view programName;
    uvec3 localSize{};
};

struct Dispatch {
    std::string_view programName;
    uvec3 localSize{};
    uvec3 workGroups{};
    // Host execution: each worker thread calls makeWorker() once and gets a
    // function that runs whole workgroups on its own copy of the kernel, so
    // Shared variables of concurrently running workgroups stay apart.
    std::function<std::function<void(uvec3)>()> makeWorker;
};

namespace detail {

template <class T>
struct ScalarKindOf;
template <>
struct ScalarKindOf<std::int32_t> : std::integral_constant<ScalarKind, ScalarKind::Int> {};
template <>
struct ScalarKindOf<std::uint32_t> : std::integral_constant<ScalarKind, ScalarKind::UInt> {};
template <>
struct ScalarKindOf<float> : std::integral_constant<ScalarKind, ScalarKind::Float> {};
template <>
struct ScalarKindOf<double> : std::integral_constant<ScalarKind, ScalarKind::Double> {};

// Scalar type and shape of a uniform value: scalar, vecN or matCxR.
template <class T>
struct UniformShape {
    using Scalar = T;
    static constexpr std::uint32_t components = 1;
    static constexpr std::uint32_t columns = 1;
};
template <class T, int N>
struct UniformShape<vec<T, N>> {
    using Scalar = T;
    static constexpr std::uint32_t components = N;
    static constexpr std::uint32_t columns = 1;
};
template <class T, int C, int R>
struct UniformShape<mat<T, C, R>> {
    using Scalar = T;
    static constexpr std::uint32_t components = R;
    static constexpr std::uint32_t columns = C;
};

template <class I>
inline constexpr bool isImageLike = IsImageBinding<I>::value or IsTextureBinding<I>::value;

struct WorkGroupPosition {
    uvec3 groupId{};
    uvec3 localSize{};
    uvec3 workGroups{};
};

// Sets the GLSL built-ins of invocation `index` of a workgroup.
inline void enterInvocation(WorkGroupPosition const &position, uint const index) noexcept {
    auto const size = position.localSize;
    gl_NumWorkGroups = position.workGroups;
    gl_WorkGroupSize = size;
    gl_WorkGroupID = position.groupId;
    gl_LocalInvocationID = uvec3{index % size.x, index / size.x % size.y, index / (size.x * size.y)};
    gl_GlobalInvocationID = position.groupId * size + gl_LocalInvocationID;
    gl_LocalInvocationIndex = index;
}

// Emulates one workgroup on this thread. Without barriers the invocations
// simply run one after another; with barriers they run as fibers that wait
// for each other at every barrier().
template <class K>
void runWorkGroup(K &kernel, WorkGroupPosition const &position) {
    auto const count = position.localSize.x * position.localSize.y * position.localSize.z;
    if constexpr (UsesBarrier<K>::value) {
        struct Context {
            K *kernel;
            WorkGroupPosition position;
        };
        auto context = Context{&kernel, position};
        auto group = CooperativeGroup{};
        group.context = &context;
        group.prepare = [](void *const c, std::uint32_t const index) {
            enterInvocation(static_cast<Context *>(c)->position, index);
        };
        group.run = [](void *const c, std::uint32_t) { static_cast<Context *>(c)->kernel->main(); };
        runCooperative(count, group);
        flushTables(false);
    } else {
        for (uint index = 0; index < count; ++index) {
            enterInvocation(position, index);
            kernel.main();
            flushTables(false);
        }
    }
}

template <class Binding>
auto &requireAttached(Binding const &binding) {
    auto *resource = binding.attachedResource();
    if (resource == nullptr) {
        throw DeviceError{"binding " + std::to_string(Binding::binding) + " has no attached resource"};
    }
    return *resource;
}

} // namespace detail

// A compute device. The public template API mirrors the talk:
//
//   device.uploadBuffer(a);            device.useKernel(kernel);
//   device.bindBuffer(kernel.A);       device.setUniform(kernel.pad);
//   device.execute(kernel, groups);    device.downloadBuffer(c);
//
// Backends implement only the protected type-erased primitives.
struct Device {
    virtual ~Device() = default;

    virtual DeviceInfo info() const = 0;

    // A buffer the CPU and this device share without copies when the device
    // can map its memory (Vulkan; OpenGL with buffer storage). uploadBuffer
    // and downloadBuffer then only synchronise. Otherwise an ordinary host
    // buffer. Contents start zeroed.
    template <class T, Dim D = Dim::D1>
    BufferResource<T, D> createBuffer(typename DimTraits<D>::IndexType const extent) {
        auto identity = ResourceIdentity::create();
        auto const bytes = static_cast<std::size_t>(DimTraits<D>::product(extent)) * sizeof(T);
        if (auto memory = doAllocateBuffer(identity, bytes)) {
            return BufferResource<T, D>::wrap(extent, std::move(*memory), std::move(identity));
        }
        return BufferResource<T, D>{extent};
    }

    template <class T, Dim D>
    void uploadBuffer(BufferResource<T, D> const &resource) {
        doUploadBuffer(resource.identity(), resource.bytes());
    }

    template <class T, Dim D>
    void downloadBuffer(BufferResource<T, D> &resource) {
        doDownloadBuffer(resource.identity(), resource.bytes());
    }

    // A table whose words live in createBuffer() memory.
    template <class T, class Layout = Aos>
    TableResource<T, Layout> createTable(std::size_t const count) {
        auto const words = static_cast<std::int32_t>(TableResource<T, Layout>::wordsFor(count));
        return TableResource<T, Layout>{createBuffer<std::uint32_t>(words), count};
    }

    template <class T, class Layout>
    void uploadTable(TableResource<T, Layout> const &table) {
        uploadBuffer(table.words());
    }

    template <class T, class Layout>
    void downloadTable(TableResource<T, Layout> &table) {
        downloadBuffer(table.words());
    }

    // Works for ImageBinding and TextureBinding: both describe the GPU format.
    template <class I, std::enable_if_t<detail::isImageLike<I>, int> = 0>
    void uploadImage(I const &image) {
        auto &resource = detail::requireAttached(image);
        auto transfer = imageTransfer<I>(resource);
        transfer.writeTexels = [&resource](ByteSpan const texels) { writeStorageTexels<I>(resource, texels); };
        doUploadImage(transfer);
    }

    template <class I, std::enable_if_t<detail::isImageLike<I>, int> = 0>
    void downloadImage(I const &image) {
        auto &resource = detail::requireAttached(image);
        auto transfer = imageTransfer<I>(resource);
        transfer.readTexels = [&resource](ConstByteSpan const texels) { readStorageTexels<I>(resource, texels); };
        doDownloadImage(transfer);
    }

    template <class K>
    void useKernel(K const &kernel) {
        static_assert(isKernel<K>, "K must declare fileLocation, local_size and main()");
        auto use = KernelUse{};
        use.programName = K::fileLocation;
        use.localSize = uvec3{kernel.local_size};
        doUseKernel(use);
    }

    template <class T, std::uint32_t B>
    void bindBuffer(BufferBinding<T, B> const &binding) {
        auto request = BufferBindRequest{};
        request.binding = B;
        request.identity = &detail::requireAttached(binding).identity();
        doBindBuffer(request);
    }

    template <class T, std::uint32_t B, class Layout>
    void bindTable(TableBinding<T, B, Layout> const &binding) {
        auto request = BufferBindRequest{};
        request.binding = B;
        request.identity = &detail::requireAttached(binding).identity();
        doBindBuffer(request);
    }

    template <class I, std::enable_if_t<IsImageBinding<I>::value, int> = 0>
    void bindImage(I const &image) {
        auto request = ImageBindRequest{};
        request.binding = I::binding;
        request.identity = &detail::requireAttached(image).identity();
        request.format = I::Traits::format;
        doBindImage(request);
    }

    template <class I, std::enable_if_t<IsTextureBinding<I>::value, int> = 0>
    void bindTexture(I const &texture) {
        auto request = TextureBindRequest{};
        request.binding = I::binding;
        request.identity = &detail::requireAttached(texture).identity();
        request.format = I::Traits::format;
        request.filter = I::filter;
        request.wrap = I::wrap;
        doBindTexture(request);
    }

    // Binds the frame as it is now; bind again after the producer writes a
    // new frame into the same platform buffer.
    template <std::uint32_t B>
    void bindExternalTexture(ExternalTextureBinding<B> const &texture) {
        auto *image = texture.attachedImage();
        if (image == nullptr) {
            throw DeviceError{"binding " + std::to_string(B) + " has no attached external image"};
        }
        doBindExternalTexture(ExternalTextureBindRequest{B, image});
    }

    template <class T, std::uint32_t L>
    void setUniform(Uniform<T, L> const &uniform) {
        using Shape = detail::UniformShape<T>;
        using Scalar = typename Shape::Scalar;
        auto write = UniformWrite{};
        write.location = L;
        write.scalar = detail::ScalarKindOf<Scalar>::value;
        write.components = Shape::components;
        write.columns = Shape::columns;
        write.columnStride = Shape::columns == 1 ? Shape::components * sizeof(Scalar) : sizeof(T) / Shape::columns;
        write.bytes = ConstByteSpan{reinterpret_cast<std::byte const *>(&uniform.value),
                                    (write.columns - 1) * write.columnStride + write.components * sizeof(Scalar)};
        doSetUniform(write);
    }

    // Waits until all queued work has finished and its results are visible
    // to the CPU. GPU backends queue execute() calls; downloads wait on their
    // own, but reading createBuffer() memory in place needs finish() first,
    // and so does writing such memory while queued work still reads it.
    void finish() { doFinish(); }

    // While enabled, every execute() is timed on the device. Throws
    // DeviceError when info().timers is false.
    void enableTimers(bool const enabled) {
        if (enabled and not info().timers) {
            throw DeviceError{"enableTimers: " + info().backend + " device '" + info().deviceName
                              + "' has no timer queries"};
        }
        timersOn = enabled;
    }

    // Waits for queued work and returns the timings of the executes recorded
    // since the previous call, in execution order.
    std::vector<KernelTiming> takeTimings() {
        doFinish();
        return doTakeTimings();
    }

    // Queues `workGroups` workgroups of kernel.local_size invocations each.
    template <class K>
    void execute(K &kernel, uvec3 const workGroups) {
        static_assert(isKernel<K>, "K must declare fileLocation, local_size and main()");
        auto const localSize = uvec3{kernel.local_size};
        auto dispatch = Dispatch{};
        dispatch.programName = K::fileLocation;
        dispatch.localSize = localSize;
        dispatch.workGroups = workGroups;
        dispatch.makeWorker = [&kernel, localSize, workGroups]() -> std::function<void(uvec3)> {
            return [copy = kernel, localSize, workGroups](uvec3 const groupId) mutable {
                detail::runWorkGroup(copy, detail::WorkGroupPosition{groupId, localSize, workGroups});
            };
        };
        doExecute(dispatch);
    }

protected:
    // Allocates `bytes` of device memory mapped for the CPU and registers it
    // as the device mirror of `identity`; nullopt when the device cannot.
    virtual std::optional<ExternalMemory> doAllocateBuffer(ResourceIdentity const &identity, std::size_t bytes) = 0;
    virtual void doUploadBuffer(ResourceIdentity const &identity, ConstByteSpan bytes) = 0;
    virtual void doDownloadBuffer(ResourceIdentity const &identity, ByteSpan bytes) = 0;
    virtual void doUploadImage(ImageTransfer const &transfer) = 0;
    virtual void doDownloadImage(ImageTransfer const &transfer) = 0;
    virtual void doUseKernel(KernelUse const &use) = 0;
    virtual void doBindBuffer(BufferBindRequest const &request) = 0;
    virtual void doBindImage(ImageBindRequest const &request) = 0;
    virtual void doBindTexture(TextureBindRequest const &request) = 0;
    virtual void doBindExternalTexture(ExternalTextureBindRequest const &request) = 0;
    virtual void doSetUniform(UniformWrite const &write) = 0;
    virtual void doExecute(Dispatch const &dispatch) = 0;
    virtual void doFinish() = 0;
    // Called after doFinish(): every timed execute has completed.
    virtual std::vector<KernelTiming> doTakeTimings() = 0;

    bool timersEnabled() const noexcept { return timersOn; }

private:
    bool timersOn = false;

    template <class I>
    static ImageTransfer imageTransfer(typename I::Resource const &resource) {
        auto transfer = ImageTransfer{};
        transfer.identity = &resource.identity();
        transfer.format = I::Traits::format;
        transfer.dim = I::dim;
        transfer.extent = DimTraits<I::dim>::toExtent3(resource.extent());
        return transfer;
    }

    template <class I>
    static void writeStorageTexels(typename I::Resource const &resource, ByteSpan const texels) {
        using Storage = typename I::Traits::StoragePixel;
        static_assert(std::is_trivially_copyable_v<Storage>);
        if (texels.size() != resource.size() * sizeof(Storage)) {
            throw DeviceError{"image upload size mismatch"};
        }
        if constexpr (std::is_same_v<Storage, typename I::Pixel>) {
            std::memcpy(texels.data(), resource.data(), texels.size());
        } else {
            for (std::size_t i = 0; i < resource.size(); ++i) {
                auto const texel = convertPixel<Storage>(resource.data()[i]);
                std::memcpy(texels.data() + i * sizeof(Storage), &texel, sizeof(Storage));
            }
        }
    }

    template <class I>
    static void readStorageTexels(typename I::Resource &resource, ConstByteSpan const texels) {
        using Storage = typename I::Traits::StoragePixel;
        using Pixel = typename I::Pixel;
        if (texels.size() != resource.size() * sizeof(Storage)) {
            throw DeviceError{"image download size mismatch"};
        }
        if constexpr (std::is_same_v<Storage, Pixel>) {
            std::memcpy(resource.data(), texels.data(), texels.size());
        } else {
            for (std::size_t i = 0; i < resource.size(); ++i) {
                auto texel = Storage{};
                std::memcpy(&texel, texels.data() + i * sizeof(Storage), sizeof(Storage));
                resource.data()[i] = convertPixel<Pixel>(texel);
            }
        }
    }
};

} // namespace tc
