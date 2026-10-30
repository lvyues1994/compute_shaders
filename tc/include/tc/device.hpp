#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <tc/buffer.hpp>
#include <tc/builtins.hpp>
#include <tc/compressed.hpp>
#include <tc/external_image.hpp>
#include <tc/image.hpp>
#include <tc/kernel.hpp>
#include <tc/mat.hpp>
#include <tc/program.hpp>
#include <tc/span.hpp>
#include <tc/subgroup.hpp>
#include <tc/sync_fence.hpp>
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

// What one kernel may use on a device. Vulkan and OpenGL ES 3.1 guarantee at
// least 4 storage buffers, 4 storage images, 16 sampled textures, 2^27-byte
// storage buffers, 16 KiB of shared memory, 128 invocations per workgroup
// and (Vulkan) 128 bytes of push constants; offsets of buffer ranges are
// aligned to at most 256 bytes. Images may be 2048 (OpenGL ES) or 4096
// (Vulkan) texels wide, 256 deep, and have 256 array layers.
struct DeviceLimits {
    std::uint32_t storageBuffers = 0;  // buffers and tables of one kernel
    std::uint32_t storageImages = 0;
    std::uint32_t sampledTextures = 0; // textures and external textures
    std::uint64_t storageBufferBytes = 0;
    std::uint32_t sharedBytes = 0;
    std::uint32_t workGroupInvocations = 0;
    uvec3 workGroupSize{};
    uvec3 workGroupCount{};
    // Bytes of Uniform values; nullopt where uniforms are not push constants
    // (OpenGL), so their size is not limited this way.
    std::optional<std::uint32_t> pushConstantBytes;
    // BufferBinding ranges must start at a multiple of this many bytes.
    std::uint32_t storageBufferOffsetAlignment = 1;
    // Optional features (tc::KernelFeature bits) kernels may use here.
    std::uint32_t features = 0;
    // Invocations of one subgroup; 0 without subgroup support.
    std::uint32_t subgroupSize = 0;
    // CompressedFormat bits (compressedBit) the device samples, and those it
    // also samples in 2D arrays (Adreno 830's OpenGL ES: all but BC4 and BC5).
    std::uint32_t compressedFormats = 0;
    std::uint32_t compressedArrayFormats = 0;
    std::uint32_t imageDimension2D = 0; // width and height of 2D images and array layers
    std::uint32_t imageDimension3D = 0; // width, height and depth of 3D images
    std::uint32_t imageArrayLayers = 0;
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
    // submitWithFence() exports sync files and waitFor() waits on the
    // device; otherwise both wait on the host.
    bool syncFiles = false;
    DeviceLimits limits;
};

// Why `program` cannot run within `limits`, or nullopt when it fits.
std::optional<std::string> exceedsLimits(KernelProgram const &program, DeviceLimits const &limits);

// Part of a buffer, in bytes.
struct ByteRange {
    std::size_t offset = 0;
    std::size_t size = 0;
};

// The completion of the work one Device::submit() sent to the device. A
// value: copies refer to the same work, and it may outlive the device. Any
// thread may query or wait.
struct Submission {
    // What a backend provides.
    struct Completion {
        virtual ~Completion() = default;
        virtual bool isComplete() const = 0;
        virtual void wait() const = 0;
    };

    Submission() = default; // already complete
    explicit Submission(std::shared_ptr<Completion const> completion_) noexcept : completion(std::move(completion_)) {}

    bool isComplete() const { return completion == nullptr or completion->isComplete(); }
    void wait() const {
        if (completion != nullptr) {
            completion->wait();
        }
    }
    // Counts the submissions of one device from 1; 0 for a default Submission.
    std::uint64_t sequence() const noexcept { return number; }

private:
    friend struct Device;

    std::shared_ptr<Completion const> completion;
    std::uint64_t number = 0;
};

// What Device::submitWithFence() returns: the submission for the host, and
// the fence for the next user of the external images it writes.
struct FencedSubmission {
    Submission submission;
    // Empty when the device cannot export one; the work has then finished.
    SyncFence fence;
};

// Device time of one execute() call.
struct KernelTiming {
    std::string program;
    // Zero for indirect dispatches on GPU backends: the counts live on the GPU.
    uvec3 workGroups{};
    // nullopt when the GPU reported a disjoint event (a clock or power state
    // change) that invalidates the measurement.
    std::optional<double> milliseconds;
    bool indirect = false;
    // Submission::sequence() of the submission the dispatch was part of.
    std::uint64_t submission = 0;
};

// Type-erased requests passed from the template API to a backend.
struct ImageTransfer {
    ResourceIdentity const *identity = nullptr;
    ImageFormat format{};
    Dim dim = Dim::D2;
    ivec3 extent{};               // of level 0; array layers in z
    std::uint32_t levels = 1;     // mip levels of the image
    std::uint32_t firstLevel = 0; // the levels this transfer moves
    std::uint32_t levelCount = 1;
    std::function<void(std::uint32_t level, ByteSpan)> writeTexels;     // upload: fill GPU-format texels
    std::function<void(std::uint32_t level, ConstByteSpan)> readTexels; // download: consume GPU-format texels
};

struct BufferBindRequest {
    std::uint32_t binding = 0;
    ResourceIdentity const *identity = nullptr;
    std::optional<ByteRange> range; // nullopt: the whole buffer
};

struct ImageBindRequest {
    std::uint32_t binding = 0;
    ResourceIdentity const *identity = nullptr;
    ImageFormat format{};
    std::uint32_t level = 0; // the mip level kernels access
};

struct TextureBindRequest {
    std::uint32_t binding = 0;
    ResourceIdentity const *identity = nullptr;
    ImageFormat format{};
    Filter filter = Filter::Nearest;
    Wrap wrap = Wrap::ClampToEdge;
    Mip mip = Mip::None;
    std::optional<CompressedFormat> compressed; // a CompressedTextureBinding
    Dim dim = Dim::D2;
};

struct CompressedTransfer {
    CompressedImage *image = nullptr; // its blocks per level
    ResourceIdentity const *identity = nullptr;
    CompressedFormat format{};
    Dim dim = Dim::D2; // of the binding: Dim::D2Array for array textures
    ivec2 extent{};
    std::uint32_t levels = 1;
    std::uint32_t layers = 1;
};

// A device image to allocate (Device::createImage).
struct ImageCreation {
    ResourceIdentity const *identity = nullptr;
    ImageFormat format{};                       // unused when `compressed`
    std::optional<CompressedFormat> compressed; // a DeviceCompressedImage
    Dim dim = Dim::D2;
    ivec3 extent{}; // of level 0; array layers in z
    std::uint32_t levels = 1;
};

// Device::copyBufferToImage: `bytes` bytes from byte `offset` of a device
// buffer into layers [firstLayer, firstLayer + layers) of one level of a
// device image (array layers, or depth slices of a 3D image).
struct BufferImageCopy {
    ResourceIdentity const *buffer = nullptr;
    std::size_t offset = 0;
    std::size_t bytes = 0;
    ConstByteSpan host; // the whole buffer on the host: the CPU backend's working memory
    ResourceIdentity const *image = nullptr;
    ImageFormat format{};                       // unused when `compressed`
    std::optional<CompressedFormat> compressed; // blocks for a DeviceCompressedImage
    Dim dim = Dim::D2;
    ivec3 extent{}; // of the level in texels; layers in z
    std::uint32_t level = 0;
    std::uint32_t firstLayer = 0;
    std::uint32_t layers = 1;
    // The CPU backend's image: stores `bytes` of texels in the GPU storage
    // format, or of blocks to decode.
    std::function<void(ConstByteSpan)> writeHost;
};

struct ExternalTextureBindRequest {
    std::uint32_t binding = 0;
    ExternalImage *image = nullptr;
};

struct ExternalImageBindRequest {
    std::uint32_t binding = 0;
    ExternalImage *image = nullptr;
    ImageFormat format{};
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

// Where an indirect dispatch reads its DispatchIndirectCommand.
struct IndirectArguments {
    ResourceIdentity const *identity = nullptr;
    std::size_t offset = 0;                        // bytes into the buffer
    DispatchIndirectCommand const *host = nullptr; // the host copy, the CPU backend's working memory
};

struct Dispatch {
    std::string_view programName;
    uvec3 localSize{};
    uvec3 workGroups{}; // unused when `indirect` is set
    std::optional<IndirectArguments> indirect;
    // Host execution: each worker thread calls makeWorker(workGroups) once and
    // gets a function that runs whole workgroups on its own copy of the
    // kernel, so Shared variables of concurrently running workgroups stay
    // apart.
    std::function<std::function<void(uvec3)>(uvec3 workGroups)> makeWorker;
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
    if constexpr (UsesBarrier<K>::value or UsesSubgroups<K>::value) {
        struct Context {
            K *kernel;
            WorkGroupPosition position;
        };
        auto context = Context{&kernel, position};
        auto group = CooperativeGroup{};
        group.context = &context;
        group.prepare = [](void *const c, std::uint32_t const index) {
            auto const &position = static_cast<Context *>(c)->position;
            enterInvocation(position, index);
            if constexpr (UsesSubgroups<K>::value) {
                enterSubgroup(index, position.localSize.x * position.localSize.y * position.localSize.z);
            }
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

// Device images need no host storage to be bound.
template <class Binding>
void requireImage(Binding const &binding) {
    if (binding.identity() == nullptr) {
        throw DeviceError{"binding " + std::to_string(Binding::binding) + " has no attached image"};
    }
}

// Why `extent` (array layers in z) is too large for `limits`, or nullopt.
inline std::optional<std::string> imageExceedsLimits(Dim const dim, ivec3 const extent, DeviceLimits const &limits) {
    auto const side = static_cast<std::uint32_t>(std::max(extent.x, extent.y));
    if (dim == Dim::D3) {
        auto const largest = std::max(side, static_cast<std::uint32_t>(extent.z));
        if (largest > limits.imageDimension3D) {
            return "3D images are at most " + std::to_string(limits.imageDimension3D) + " texels on a side here";
        }
        return std::nullopt;
    }
    if (side > limits.imageDimension2D) {
        return "images are at most " + std::to_string(limits.imageDimension2D) + " texels wide and high here";
    }
    if (dim == Dim::D2Array and static_cast<std::uint32_t>(extent.z) > limits.imageArrayLayers) {
        return "array images have at most " + std::to_string(limits.imageArrayLayers) + " layers here";
    }
    return std::nullopt;
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

    // An image only this device holds (see DeviceImage), with `levels` mip
    // levels (0: the full chain) and, for Dim::D2Array, `extent.z` layers.
    template <class G, Dim D, class P = typename GPUFormatTraits<G>::StoragePixel>
    DeviceImage<G, D, P> createImage(typename DimTraits<D>::IndexType const extent, std::uint32_t levels = 1) {
        static_assert(D != Dim::D1, "1D images are unavailable on OpenGL ES; use a buffer instead");
        auto const extent3 = DimTraits<D>::toExtent3(extent);
        auto const full = extent3.x > 0 and extent3.y > 0 and extent3.z > 0 ? fullMipLevels(D, extent3) : 0u;
        levels = levels == 0 ? full : levels;
        if (full == 0 or levels > full) {
            throw DeviceError{"createImage: empty extent or more mip levels than it has"};
        }
        if (auto const reason = detail::imageExceedsLimits(D, extent3, info().limits)) {
            throw DeviceError{"createImage: " + *reason};
        }
        auto state = std::make_shared<detail::DeviceImageState<P, D>>(extent, levels);
        auto request = ImageCreation{};
        request.identity = &state->identity;
        request.format = GPUFormatTraits<G>::format;
        request.dim = D;
        request.extent = extent3;
        request.levels = levels;
        if (doCreateImage(request)) {
            state->host = std::make_unique<MipChain<P, D>>(extent, levels);
        }
        return DeviceImage<G, D, P>{std::move(state)};
    }

    // A compressed image only this device holds (see DeviceCompressedImage),
    // with `levels` mip levels (0: the full chain) and, for Dim::D2Array,
    // `extent.z` layers. Throws DeviceError when the device does not sample
    // `format` (DeviceLimits::compressedFormats).
    template <Dim D>
    DeviceCompressedImage<D> createImage(CompressedFormat const format, typename DimTraits<D>::IndexType const extent,
                                         std::uint32_t levels = 1) {
        static_assert(D == Dim::D2 or D == Dim::D2Array, "compressed images are 2D or 2D arrays");
        auto const extent3 = DimTraits<D>::toExtent3(extent);
        auto const full = extent3.x > 0 and extent3.y > 0 and extent3.z > 0 ? fullMipLevels(Dim::D2, extent3) : 0u;
        levels = levels == 0 ? full : levels;
        if (full == 0 or levels > full) {
            throw DeviceError{"createImage: empty extent or more mip levels than it has"};
        }
        auto const limits = info().limits;
        if ((limits.compressedFormats & compressedBit(format)) == 0) {
            throw DeviceError{std::string{"createImage: this device does not sample "}
                              + compressedFormatInfo(format).name + " textures"};
        }
        if (D == Dim::D2Array and (limits.compressedArrayFormats & compressedBit(format)) == 0) {
            throw DeviceError{std::string{"createImage: this device has no "} + compressedFormatInfo(format).name
                              + " array textures"};
        }
        if (auto const reason = detail::imageExceedsLimits(D, extent3, limits)) {
            throw DeviceError{"createImage: " + *reason};
        }
        auto const layers = static_cast<std::uint32_t>(extent3.z);
        auto state =
            std::make_shared<detail::DeviceCompressedState>(format, ivec2{extent3.x, extent3.y}, levels, layers);
        auto request = ImageCreation{};
        request.identity = &state->identity;
        request.compressed = format;
        request.dim = D;
        request.extent = extent3;
        request.levels = levels;
        if (doCreateImage(request)) {
            state->host = std::make_unique<DecodedChain>(extent3, levels);
        }
        return DeviceCompressedImage<D>{std::move(state)};
    }

    // Copies texels from a buffer on the device (uploaded, or from
    // createBuffer) into layers [firstLayer, firstLayer + layers) of mip
    // level `level` of `image` (array layers, or depth slices of a 3D
    // image), in order with the work around it and without the host. From
    // byte `offset`, a multiple of 4 and of the texel size, the buffer holds
    // the texels in the GPU storage format (GPUFormatTraits<G>::StoragePixel)
    // row after row and layer after layer.
    template <class T, Dim BD, class G, Dim D, class P>
    void copyBufferToImage(BufferResource<T, BD> const &buffer, std::size_t const offset,
                           DeviceImage<G, D, P> const &image, std::uint32_t const level,
                           std::uint32_t const firstLayer = 0, std::uint32_t const layers = 1) {
        using Storage = typename GPUFormatTraits<G>::StoragePixel;
        if (not image or level >= image.levels()) {
            throw DeviceError{"copyBufferToImage: no image, or no mip level " + std::to_string(level)};
        }
        auto const extent = DimTraits<D>::toExtent3(image.extent(level));
        auto const depth = static_cast<std::uint32_t>(extent.z);
        if (layers == 0 or firstLayer >= depth or layers > depth - firstLayer) {
            throw DeviceError{"copyBufferToImage: layers " + std::to_string(firstLayer) + " + " + std::to_string(layers)
                              + " are outside the image, which has " + std::to_string(depth)};
        }
        auto const bytes = static_cast<std::size_t>(extent.x) * static_cast<std::size_t>(extent.y) * layers * sizeof(Storage);
        auto const host = buffer.bytes();
        if (offset % 4 != 0 or offset % sizeof(Storage) != 0 or offset > host.size() or bytes > host.size() - offset) {
            throw DeviceError{"copyBufferToImage: " + std::to_string(bytes) + " bytes from byte " + std::to_string(offset)
                              + " are outside the buffer or misaligned"};
        }
        auto copy = BufferImageCopy{};
        copy.buffer = &buffer.identity();
        copy.offset = offset;
        copy.bytes = bytes;
        copy.host = host;
        copy.image = &image.identity();
        copy.format = GPUFormatTraits<G>::format;
        copy.dim = D;
        copy.extent = extent;
        copy.level = level;
        copy.firstLayer = firstLayer;
        copy.layers = layers;
        if (auto *chain = image.shared()->host.get()) {
            auto *first = chain->level(level).data()
                        + static_cast<std::size_t>(firstLayer) * static_cast<std::size_t>(extent.x * extent.y);
            copy.writeHost = [first](ConstByteSpan const texels) {
                for (std::size_t i = 0; i < texels.size() / sizeof(Storage); ++i) {
                    auto texel = Storage{};
                    std::memcpy(&texel, texels.data() + i * sizeof(Storage), sizeof(Storage));
                    first[i] = convertPixel<P>(texel);
                }
            };
        }
        doCopyBufferToImage(copy);
    }

    // The same for a compressed image: from byte `offset`, a multiple of 4
    // and of the block size, the buffer holds the blocks of the layers in
    // turn (DeviceCompressedImage::layerBytes(level) bytes each), each layer
    // in row-major order as in a KTX2 file.
    template <class T, Dim BD, Dim D>
    void copyBufferToImage(BufferResource<T, BD> const &buffer, std::size_t const offset,
                           DeviceCompressedImage<D> const &image, std::uint32_t const level,
                           std::uint32_t const firstLayer = 0, std::uint32_t const layers = 1) {
        if (not image or level >= image.levels()) {
            throw DeviceError{"copyBufferToImage: no image, or no mip level " + std::to_string(level)};
        }
        if (layers == 0 or firstLayer >= image.layers() or layers > image.layers() - firstLayer) {
            throw DeviceError{"copyBufferToImage: layers " + std::to_string(firstLayer) + " + " + std::to_string(layers)
                              + " are outside the image, which has " + std::to_string(image.layers())};
        }
        auto const blockBytes = compressedFormatInfo(image.format()).blockBytes;
        auto const layerBytes = image.layerBytes(level);
        auto const bytes = layerBytes * layers;
        auto const host = buffer.bytes();
        if (offset % 4 != 0 or offset % blockBytes != 0 or offset > host.size() or bytes > host.size() - offset) {
            throw DeviceError{"copyBufferToImage: " + std::to_string(bytes) + " bytes from byte " + std::to_string(offset)
                              + " are outside the buffer or misaligned"};
        }
        auto const size = DimTraits<D>::toExtent3(image.extent(level));
        auto copy = BufferImageCopy{};
        copy.buffer = &buffer.identity();
        copy.offset = offset;
        copy.bytes = bytes;
        copy.host = host;
        copy.image = &image.identity();
        copy.compressed = image.format();
        copy.dim = D;
        copy.extent = ivec3{size.x, size.y, static_cast<std::int32_t>(image.layers())};
        copy.level = level;
        copy.firstLayer = firstLayer;
        copy.layers = layers;
        if (auto *chain = image.shared()->host.get()) {
            auto *first = chain->level(level).data() + static_cast<std::size_t>(firstLayer) * size.x * size.y;
            copy.writeHost = [first, size, layerBytes, format = image.format()](ConstByteSpan const blocks) {
                auto const *data = reinterpret_cast<std::uint8_t const *>(blocks.data());
                for (std::size_t layer = 0; layer * layerBytes < blocks.size(); ++layer) {
                    decodeCompressed(format, ivec2{size.x, size.y}, data + layer * layerBytes, layerBytes,
                                     first + layer * static_cast<std::size_t>(size.x * size.y));
                }
            };
        }
        doCopyBufferToImage(copy);
    }

    template <class T, Dim D>
    void uploadBuffer(BufferResource<T, D> const &resource) {
        doUploadBuffer(resource.identity(), resource.bytes(), ByteRange{0, resource.bytes().size()});
    }

    // Uploads only elements [first, first + count). A buffer the device has
    // never seen is uploaded whole.
    template <class T, Dim D>
    void uploadBuffer(BufferResource<T, D> const &resource, std::size_t const first, std::size_t const count) {
        doUploadBuffer(resource.identity(), resource.bytes(), elementRange<T>(resource.size(), first, count, "upload"));
    }

    template <class T, Dim D>
    void downloadBuffer(BufferResource<T, D> &resource) {
        doDownloadBuffer(resource.identity(), resource.bytes(), ByteRange{0, resource.bytes().size()});
    }

    // Downloads only elements [first, first + count); the rest of the host
    // copy keeps its contents.
    template <class T, Dim D>
    void downloadBuffer(BufferResource<T, D> &resource, std::size_t const first, std::size_t const count) {
        doDownloadBuffer(resource.identity(), resource.bytes(), elementRange<T>(resource.size(), first, count, "download"));
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

    // Uploads the words of elements [first, first + count) (see
    // TableResource::wordsOf); the element count must not have changed.
    template <class T, class Layout>
    void uploadTable(TableResource<T, Layout> const &table, std::size_t const first, std::size_t const count) {
        auto const words = tableWords(table, first, count, "uploadTable");
        uploadBuffer(table.words(), words.first, words.count);
    }

    template <class T, class Layout>
    void downloadTable(TableResource<T, Layout> &table) {
        downloadBuffer(table.words());
    }

    template <class T, class Layout>
    void downloadTable(TableResource<T, Layout> &table, std::size_t const first, std::size_t const count) {
        auto const words = tableWords(table, first, count, "downloadTable");
        downloadBuffer(table.words(), words.first, words.count);
    }

    // Works for ImageBinding and TextureBinding: both describe the GPU
    // format. Moves every mip level of an attached MipChain.
    template <class I, std::enable_if_t<detail::isImageLike<I>, int> = 0>
    void uploadImage(I const &image) {
        uploadLevels(image, 0, imageLevels(image));
    }

    // Uploads one mip level; an image the device has never seen is
    // uploaded whole.
    template <class I, std::enable_if_t<detail::isImageLike<I>, int> = 0>
    void uploadImage(I const &image, std::uint32_t const level) {
        uploadLevels(image, level, 1);
    }

    // Compressed textures are uploaded whole and only sampled. Throws
    // DeviceError when the device does not sample the format (see
    // DeviceLimits::compressedFormats).
    template <std::uint32_t B, Filter F, Wrap W, Mip M, Dim D>
    void uploadImage(CompressedTextureBinding<B, F, W, M, D> const &texture) {
        auto *image = texture.attachedImage();
        if (image == nullptr) {
            throw DeviceError{"binding " + std::to_string(B)
                              + (texture.attachedDeviceImage() != nullptr
                                     ? " is attached to a device image, which has no host copy; fill it with "
                                       "copyBufferToImage"
                                     : " has no attached compressed image")};
        }
        if (D == Dim::D2 and image->layers() != 1) {
            throw DeviceError{"uploadImage: binding " + std::to_string(B) + " is 2D, but the image has "
                              + std::to_string(image->layers()) + " layers; use Dim::D2Array"};
        }
        if (D == Dim::D2Array and (info().limits.compressedArrayFormats & compressedBit(image->format())) == 0) {
            throw DeviceError{std::string{"uploadImage: this device has no "} + compressedFormatInfo(image->format()).name
                              + " array textures"};
        }
        doUploadCompressed(CompressedTransfer{image, &image->identity(), image->format(), D, image->extent(),
                                              image->levels(), image->layers()});
    }

    template <class I, std::enable_if_t<detail::isImageLike<I>, int> = 0>
    void downloadImage(I const &image) {
        downloadLevels(image, 0, imageLevels(image));
    }

    template <class I, std::enable_if_t<detail::isImageLike<I>, int> = 0>
    void downloadImage(I const &image, std::uint32_t const level) {
        downloadLevels(image, level, 1);
    }

    template <class K>
    void useKernel(K const &kernel) {
        static_assert(isKernel<K>, "K must declare fileLocation, local_size and main()");
        auto use = KernelUse{};
        use.programName = K::fileLocation;
        use.localSize = uvec3{kernel.local_size};
        doUseKernel(use);
    }

    // Binds the resource, or the range given to BufferBinding::attach.
    template <class T, std::uint32_t B>
    void bindBuffer(BufferBinding<T, B> const &binding) {
        auto request = BufferBindRequest{};
        request.binding = B;
        request.identity = &detail::requireAttached(binding).identity();
        if (binding.isRange()) {
            if (binding.elementCount() == 0) {
                throw DeviceError{"bindBuffer: binding " + std::to_string(B) + " is an empty range"};
            }
            request.range = ByteRange{binding.firstElement() * sizeof(T), binding.elementCount() * sizeof(T)};
        }
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
        detail::requireImage(image);
        auto request = ImageBindRequest{};
        request.binding = I::binding;
        request.identity = image.identity();
        request.format = I::Traits::format;
        request.level = image.level();
        doBindImage(request);
    }

    template <class I, std::enable_if_t<IsTextureBinding<I>::value, int> = 0>
    void bindTexture(I const &texture) {
        detail::requireImage(texture);
        auto request = TextureBindRequest{};
        request.binding = I::binding;
        request.identity = texture.identity();
        request.format = I::Traits::format;
        request.filter = I::filter;
        request.wrap = I::wrap;
        request.mip = I::mip;
        request.dim = I::dim;
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

    template <std::uint32_t B, Filter F, Wrap W, Mip M, Dim D>
    void bindTexture(CompressedTextureBinding<B, F, W, M, D> const &texture) {
        auto request = TextureBindRequest{};
        request.binding = B;
        if (auto const *image = texture.attachedImage()) {
            request.identity = &image->identity();
            request.compressed = image->format();
        } else if (auto const *device = texture.attachedDeviceImage()) {
            request.identity = &device->identity;
            request.compressed = device->format;
        } else {
            throw DeviceError{"binding " + std::to_string(B) + " has no attached compressed image"};
        }
        request.filter = F;
        request.wrap = W;
        request.mip = M;
        request.dim = D;
        doBindTexture(request);
    }

    // Binds an RGBA frame as a storage image. The frame's next user must
    // wait for the work that writes it: pass it the fence of
    // submitWithFence().
    template <class G, std::uint32_t B>
    void bindExternalImage(ExternalImageBinding<G, B> const &image) {
        auto *frame = image.attachedImage();
        if (frame == nullptr) {
            throw DeviceError{"binding " + std::to_string(B) + " has no attached external image"};
        }
        auto const format = GPUFormatTraits<G>::format;
        if (not frame->storageFormat() or not(*frame->storageFormat() == format)) {
            throw DeviceError{"bindExternalImage: binding " + std::to_string(B) + " is " + glslFormatQualifier(format)
                              + ", but the frame is not an " + glslFormatQualifier(format) + " RGBA frame"};
        }
        doBindExternalImage(ExternalImageBindRequest{B, frame, format});
    }

    // Work queued after this call waits on the device (or, where the device
    // cannot, on the host) for `fence`: e.g. the release fence of a frame
    // the compositor gives back (ASurfaceTransactionStats_getPreviousReleaseFenceFd).
    void waitFor(SyncFence fence) { doWaitFor(std::move(fence)); }

    // Submits the queued work like submit() and also returns a fence the
    // device signals when it is done, for the next user of the external
    // images it writes (the acquire fence of ASurfaceTransaction_setBuffer).
    FencedSubmission submitWithFence() {
        auto result = doSubmitWithFence();
        result.submission.number = ++submitted;
        return result;
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

    // Why the transpiled kernel `programName` cannot run on this device
    // (missing, over a limit, needing a feature or an API variant the device
    // lacks), or nullopt when it can. Checking every program of
    // ProgramRegistry::all() at startup finds problems before any work runs.
    std::optional<std::string> checkKernel(std::string_view const programName) const {
        auto const program = ProgramRegistry::instance().find(programName);
        if (not program) {
            return "kernel '" + std::string{programName} + "' was not transpiled; add its header with tc_add_kernels()";
        }
        if (auto reason = exceedsLimits(*program, info().limits)) {
            return reason;
        }
        return doCheckKernel(*program);
    }

    // Sends the work queued so far to the device and returns without waiting;
    // later work goes into the next submission. The host can prepare the
    // next frame while the device runs this one, but must not write memory
    // the submission still reads (createBuffer() memory: use one slice of a
    // ring buffer per frame in flight). Once the submission is complete,
    // createBuffer() memory holds its results, so it serves as asynchronous
    // readback; downloadBuffer() and downloadImage() always wait.
    Submission submit() {
        auto submission = doSubmit();
        submission.number = ++submitted;
        return submission;
    }

    // Waits until all queued work has finished and its results are visible
    // to the CPU. GPU backends queue execute() calls; downloads wait on their
    // own, but reading createBuffer() memory in place needs finish() (or a
    // completed Submission) first, and so does writing such memory while
    // queued work still reads it.
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

    // Waits for all queued work, submitted or not, and returns the timings
    // of every execute not taken yet, in execution order.
    std::vector<KernelTiming> takeTimings() {
        doFinish();
        return doTakeTimings(std::numeric_limits<std::uint64_t>::max());
    }

    // Waits for `submission` only and returns the timings of the executes
    // it and earlier submissions of this device contain that were not taken
    // yet. Later work keeps running and recording, so a frame loop with
    // several frames in flight takes the timings of a frame it has already
    // waited for, e.g. before reusing its slot.
    std::vector<KernelTiming> takeTimings(Submission const &submission) {
        submission.wait();
        return doTakeTimings(submission.sequence());
    }

    // Queues `workGroups` workgroups of kernel.local_size invocations each.
    template <class K>
    void execute(K &kernel, uvec3 const workGroups) {
        auto dispatch = makeDispatch(kernel);
        dispatch.workGroups = workGroups;
        doExecute(dispatch);
    }

    // Queues the kernel with the workgroup counts arguments[index] holds when
    // the device reaches this dispatch, typically written by an earlier
    // kernel. The buffer must be on the device (uploadBuffer or
    // createBuffer); counts above the device limits are undefined behaviour
    // on GPU backends.
    template <class K>
    void executeIndirect(K &kernel, BufferResource<DispatchIndirectCommand> const &arguments,
                         std::size_t const index = 0) {
        if (index >= arguments.size()) {
            throw DeviceError{"executeIndirect: argument " + std::to_string(index) + " is outside the buffer"};
        }
        auto dispatch = makeDispatch(kernel);
        dispatch.indirect =
            IndirectArguments{&arguments.identity(), index * sizeof(DispatchIndirectCommand), arguments.data() + index};
        doExecute(dispatch);
    }

protected:
    // Allocates `bytes` of device memory mapped for the CPU and registers it
    // as the device mirror of `identity`; nullopt when the device cannot.
    virtual std::optional<ExternalMemory> doAllocateBuffer(ResourceIdentity const &identity, std::size_t bytes) = 0;
    // `bytes` is the whole host buffer; only `range` of it moves.
    virtual void doUploadBuffer(ResourceIdentity const &identity, ConstByteSpan bytes, ByteRange range) = 0;
    virtual void doDownloadBuffer(ResourceIdentity const &identity, ByteSpan bytes, ByteRange range) = 0;
    // Reasons specific to the backend, after the limits passed.
    virtual std::optional<std::string> doCheckKernel(KernelProgram const &program) const {
        (void)program;
        return std::nullopt;
    }
    virtual void doUploadImage(ImageTransfer const &transfer) = 0;
    virtual void doDownloadImage(ImageTransfer const &transfer) = 0;
    // Allocates the image and registers it as the device mirror of its
    // identity; returns true when the device runs kernels on the host and
    // the image needs host storage (the CPU backend).
    virtual bool doCreateImage(ImageCreation const &request) = 0;
    virtual void doCopyBufferToImage(BufferImageCopy const &copy) = 0;
    virtual void doUseKernel(KernelUse const &use) = 0;
    virtual void doBindBuffer(BufferBindRequest const &request) = 0;
    virtual void doBindImage(ImageBindRequest const &request) = 0;
    virtual void doBindTexture(TextureBindRequest const &request) = 0;
    virtual void doBindExternalTexture(ExternalTextureBindRequest const &request) = 0;
    virtual void doBindExternalImage(ExternalImageBindRequest const &request) {
        (void)request;
        throw DeviceError{"external images are only available on Android"};
    }
    virtual void doWaitFor(SyncFence fence) { fence.wait(); }
    virtual void doUploadCompressed(CompressedTransfer const &transfer) {
        throw DeviceError{std::string{"compressed format "} + compressedFormatInfo(transfer.format).name
                          + " is not supported by this backend"};
    }
    virtual FencedSubmission doSubmitWithFence() {
        auto submission = doSubmit();
        submission.wait();
        return FencedSubmission{std::move(submission), SyncFence{}};
    }
    virtual void doSetUniform(UniformWrite const &write) = 0;
    virtual void doExecute(Dispatch const &dispatch) = 0;
    virtual Submission doSubmit() = 0;
    virtual void doFinish() = 0;
    // The timings of submissions 1 to `last`, whose work has completed;
    // later timings stay for a later call.
    virtual std::vector<KernelTiming> doTakeTimings(std::uint64_t last) = 0;

    bool timersEnabled() const noexcept { return timersOn; }
    // The Submission::sequence() that work queued now will be part of.
    std::uint64_t nextSubmission() const noexcept { return submitted + 1; }

    // Moves the timings of submissions up to `last` out of `timings`,
    // keeping their order.
    static std::vector<KernelTiming> takeThrough(std::vector<KernelTiming> &timings, std::uint64_t const last) {
        auto const later = std::stable_partition(timings.begin(), timings.end(),
                                                 [last](KernelTiming const &t) { return t.submission <= last; });
        auto taken = std::vector<KernelTiming>(std::make_move_iterator(timings.begin()), std::make_move_iterator(later));
        timings.erase(timings.begin(), later);
        return taken;
    }

private:
    bool timersOn = false;
    std::uint64_t submitted = 0;

    template <class T>
    static ByteRange elementRange(std::size_t const size, std::size_t const first, std::size_t const count,
                                  char const *what) {
        if (first > size or count > size - first) {
            throw DeviceError{std::string{what} + "Buffer: elements [" + std::to_string(first) + ", "
                              + std::to_string(first + count) + ") are outside the " + std::to_string(size)
                              + "-element buffer"};
        }
        return ByteRange{first * sizeof(T), count * sizeof(T)};
    }

    template <class T, class Layout>
    static typename TableResource<T, Layout>::WordRange tableWords(TableResource<T, Layout> const &table,
                                                                   std::size_t const first, std::size_t const count,
                                                                   char const *what) {
        if (first > table.size() or count > table.size() - first) {
            throw DeviceError{std::string{what} + ": elements [" + std::to_string(first) + ", "
                              + std::to_string(first + count) + ") are outside the table"};
        }
        return table.wordsOf(first, count);
    }

    template <class K>
    static Dispatch makeDispatch(K &kernel) {
        static_assert(isKernel<K>, "K must declare fileLocation, local_size and main()");
        auto const localSize = uvec3{kernel.local_size};
        auto dispatch = Dispatch{};
        dispatch.programName = K::fileLocation;
        dispatch.localSize = localSize;
        dispatch.makeWorker = [&kernel, localSize](uvec3 const workGroups) -> std::function<void(uvec3)> {
            return [copy = kernel, localSize, workGroups](uvec3 const groupId) mutable {
                detail::runWorkGroup(copy, detail::WorkGroupPosition{groupId, localSize, workGroups});
            };
        };
        return dispatch;
    }

    template <class I>
    static std::uint32_t imageLevels(I const &image) {
        requireHostImage(image);
        return image.levelCount();
    }

    template <class I>
    static void requireHostImage(I const &image) {
        if (image.isDeviceImage()) {
            throw DeviceError{"binding " + std::to_string(I::binding)
                              + " is attached to a device image, which has no host copy to transfer; write it with a "
                                "kernel or copyBufferToImage"};
        }
        detail::requireAttached(image);
    }

    template <class I>
    static ImageTransfer imageTransfer(I const &image, std::uint32_t const first, std::uint32_t const count) {
        requireHostImage(image);
        if (first >= image.levelCount() or count > image.levelCount() - first) {
            throw DeviceError{"image transfer: mip level " + std::to_string(first + count - 1) + " does not exist"};
        }
        auto transfer = ImageTransfer{};
        transfer.identity = image.identity();
        transfer.format = I::Traits::format;
        transfer.dim = I::dim;
        transfer.extent = DimTraits<I::dim>::toExtent3(image.levelResource(0)->extent());
        transfer.levels = image.levelCount();
        transfer.firstLevel = first;
        transfer.levelCount = count;
        return transfer;
    }

    template <class I>
    void uploadLevels(I const &image, std::uint32_t const first, std::uint32_t const count) {
        auto transfer = imageTransfer(image, first, count);
        transfer.writeTexels = [&image](std::uint32_t const level, ByteSpan const texels) {
            writeStorageTexels<I>(*image.levelResource(level), texels);
        };
        doUploadImage(transfer);
    }

    template <class I>
    void downloadLevels(I const &image, std::uint32_t const first, std::uint32_t const count) {
        auto transfer = imageTransfer(image, first, count);
        transfer.readTexels = [&image](std::uint32_t const level, ConstByteSpan const texels) {
            readStorageTexels<I>(*image.levelResource(level), texels);
        };
        doDownloadImage(transfer);
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
