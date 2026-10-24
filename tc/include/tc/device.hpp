#pragma once

#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include <tc/buffer.hpp>
#include <tc/builtins.hpp>
#include <tc/image.hpp>
#include <tc/kernel.hpp>
#include <tc/program.hpp>
#include <tc/span.hpp>
#include <tc/uniform.hpp>

namespace tc {

// Misuse of a device (unbound resource, unknown kernel) or a failing graphics
// API call. Device creation failures are reported the same way.
struct DeviceError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct DeviceInfo {
    std::string backend;
    std::string deviceName;
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

struct UniformWrite {
    std::uint32_t location = 0;
    ScalarKind scalar = ScalarKind::Float;
    std::uint32_t components = 1;
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
    std::function<void(uvec3)> runWorkGroup; // runs every invocation of one workgroup on the host
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

template <class T>
constexpr std::uint32_t componentCount() noexcept {
    if constexpr (VecTraits<T>::isVector) {
        return static_cast<std::uint32_t>(T::length());
    } else {
        return 1;
    }
}

// Emulates one workgroup: invocations run one after another on this thread.
template <class K>
void runWorkGroup(K &kernel, uvec3 const groupId, uvec3 const localSize, uvec3 const workGroups) {
    gl_NumWorkGroups = workGroups;
    gl_WorkGroupSize = localSize;
    gl_WorkGroupID = groupId;
    for (uint z = 0; z < localSize.z; ++z) {
        for (uint y = 0; y < localSize.y; ++y) {
            for (uint x = 0; x < localSize.x; ++x) {
                gl_LocalInvocationID = uvec3{x, y, z};
                gl_GlobalInvocationID = groupId * localSize + gl_LocalInvocationID;
                gl_LocalInvocationIndex = (z * localSize.y + y) * localSize.x + x;
                kernel.main();
            }
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

    template <class T, Dim D>
    void uploadBuffer(BufferResource<T, D> const &resource) {
        doUploadBuffer(resource.identity(), resource.bytes());
    }

    template <class T, Dim D>
    void downloadBuffer(BufferResource<T, D> &resource) {
        doDownloadBuffer(resource.identity(), resource.bytes());
    }

    template <class I, std::enable_if_t<IsImageBinding<I>::value, int> = 0>
    void uploadImage(I const &image) {
        auto &resource = detail::requireAttached(image);
        auto transfer = imageTransfer<I>(resource);
        transfer.writeTexels = [&resource](ByteSpan const texels) { writeStorageTexels<I>(resource, texels); };
        doUploadImage(transfer);
    }

    template <class I, std::enable_if_t<IsImageBinding<I>::value, int> = 0>
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

    template <class I, std::enable_if_t<IsImageBinding<I>::value, int> = 0>
    void bindImage(I const &image) {
        auto request = ImageBindRequest{};
        request.binding = I::binding;
        request.identity = &detail::requireAttached(image).identity();
        request.format = I::Traits::format;
        doBindImage(request);
    }

    template <class T, std::uint32_t L>
    void setUniform(Uniform<T, L> const &uniform) {
        using Component = detail::ComponentOf<T>;
        auto write = UniformWrite{};
        write.location = L;
        write.scalar = detail::ScalarKindOf<Component>::value;
        write.components = detail::componentCount<T>();
        write.bytes = ConstByteSpan{reinterpret_cast<std::byte const *>(&uniform.value),
                                    write.components * sizeof(Component)};
        doSetUniform(write);
    }

    // Runs `workGroups` workgroups of kernel.local_size invocations each.
    template <class K>
    void execute(K &kernel, uvec3 const workGroups) {
        static_assert(isKernel<K>, "K must declare fileLocation, local_size and main()");
        auto const localSize = uvec3{kernel.local_size};
        auto dispatch = Dispatch{};
        dispatch.programName = K::fileLocation;
        dispatch.localSize = localSize;
        dispatch.workGroups = workGroups;
        dispatch.runWorkGroup = [&kernel, localSize, workGroups](uvec3 const groupId) {
            detail::runWorkGroup(kernel, groupId, localSize, workGroups);
        };
        doExecute(dispatch);
    }

protected:
    virtual void doUploadBuffer(ResourceIdentity const &identity, ConstByteSpan bytes) = 0;
    virtual void doDownloadBuffer(ResourceIdentity const &identity, ByteSpan bytes) = 0;
    virtual void doUploadImage(ImageTransfer const &transfer) = 0;
    virtual void doDownloadImage(ImageTransfer const &transfer) = 0;
    virtual void doUseKernel(KernelUse const &use) = 0;
    virtual void doBindBuffer(BufferBindRequest const &request) = 0;
    virtual void doBindImage(ImageBindRequest const &request) = 0;
    virtual void doSetUniform(UniformWrite const &write) = 0;
    virtual void doExecute(Dispatch const &dispatch) = 0;

private:
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
