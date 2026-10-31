#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include <tc/buffer.hpp>
#include <tc/dim.hpp>

namespace tc {

namespace detail {

// What the copies of one MemoryHeap share.
struct HeapState {
    explicit HeapState(std::size_t const bytes_) : bytes(bytes_), identity(ResourceIdentity::create()) {}

    std::size_t bytes;
    ResourceIdentity identity;
    // The heap's memory on the CPU backend, which runs kernels on the host;
    // empty on GPU devices.
    std::shared_ptr<std::byte> host;
};

} // namespace detail

// A block of device memory that resources are placed in (Device::createHeap),
// so that resources whose lifetimes do not overlap can share memory.
// Copies refer to the same heap; it is freed once the last copy and every
// resource placed in it are gone and the device no longer uses them.
struct MemoryHeap {
    MemoryHeap() = default; // no heap

    explicit operator bool() const noexcept { return state != nullptr; }
    std::size_t size() const noexcept { return state->bytes; }
    ResourceIdentity const &identity() const noexcept { return state->identity; }

    std::shared_ptr<detail::HeapState> const &shared() const noexcept { return state; }

private:
    friend struct Device;
    template <class T, Dim D>
    friend struct DeviceBuffer;
    explicit MemoryHeap(std::shared_ptr<detail::HeapState> state_) noexcept : state(std::move(state_)) {}

    std::shared_ptr<detail::HeapState> state;
};

// Where a resource lives: `heap` from byte `offset`, a multiple of the
// alignment Device::bufferPlacement / imagePlacement give for it.
//
// Resources whose byte ranges in one heap overlap are aliases. Once one of
// them is used, the contents of the others are undefined, and a resource's
// own contents are undefined until it is first written. The device orders
// the accesses of aliases like those of one resource: the first write of
// one waits for the earlier accesses of the others.
struct Placement {
    MemoryHeap heap;
    std::size_t offset = 0;
};

// The bytes a resource takes in a heap and the alignment of its offset.
struct PlacementSize {
    std::size_t bytes = 0;
    std::size_t alignment = 1;
};

namespace detail {

// What the copies of one DeviceBuffer share.
template <class T, Dim D>
struct DeviceBufferState {
    DeviceBufferState(BufferResource<T, D> view_, std::shared_ptr<HeapState> heap_) noexcept
        : view(std::move(view_)), heap(std::move(heap_)) {}

    // The extent and identity; on the CPU backend also the elements.
    BufferResource<T, D> view;
    std::shared_ptr<HeapState> heap; // set for placed buffers: keeps the memory alive
    std::size_t offset = 0;          // in the heap
};

} // namespace detail

// A buffer that lives only on the device that created it
// (Device::createDeviceBuffer), the buffer counterpart of DeviceImage:
// there is no host copy, so it takes device memory only. Kernels read and
// write it through BufferBinding; Device::copyBuffer, copyBufferToImage and
// executeIndirect accept it. uploadBuffer and downloadBuffer do not apply.
// The contents start undefined. Copies refer to the same buffer, which is
// freed after the last copy, once the device no longer uses it.
template <class T, Dim D = Dim::D1>
struct DeviceBuffer {
    static_assert(std::is_trivially_copyable_v<T>, "GPU resources hold trivially copyable elements");

    using value_type = T;
    using IndexType = typename DimTraits<D>::IndexType;
    static constexpr Dim dim = D;

    DeviceBuffer() = default; // no buffer

    explicit operator bool() const noexcept { return state != nullptr; }
    IndexType extent() const noexcept { return state->view.extent(); }
    std::size_t size() const noexcept { return state->view.size(); }
    ResourceIdentity const &identity() const noexcept { return state->view.identity(); }
    // Where it lives in a heap; a default Placement when it has memory of its own.
    Placement placement() const;

    // What bindings attach to: the elements on the CPU backend, which keeps
    // the only copy in host memory; a view without host data (data() is
    // nullptr) on GPU devices.
    BufferResource<T, D> *hostView() const noexcept { return &state->view; }

    std::shared_ptr<detail::DeviceBufferState<T, D>> const &shared() const noexcept { return state; }

private:
    friend struct Device;
    explicit DeviceBuffer(std::shared_ptr<detail::DeviceBufferState<T, D>> state_) noexcept : state(std::move(state_)) {}

    std::shared_ptr<detail::DeviceBufferState<T, D>> state;
};

template <class T, Dim D>
Placement DeviceBuffer<T, D>::placement() const {
    if (state->heap == nullptr) {
        return Placement{};
    }
    return Placement{MemoryHeap{state->heap}, state->offset};
}

template <class T, std::uint32_t B>
void BufferBinding<T, B>::attach(DeviceBuffer<T, Dim::D1> const &buffer) noexcept {
    attach(*buffer.hostView());
}

template <class T, std::uint32_t B>
void BufferBinding<T, B>::attach(DeviceBuffer<T, Dim::D1> const &buffer, std::size_t const first_,
                                 std::size_t const count_) noexcept {
    attach(*buffer.hostView(), first_, count_);
}

namespace detail {

// The identity and host bytes of a buffer the device copies from or to:
// a BufferResource or a DeviceBuffer (whose host bytes exist only on the
// CPU backend).
struct BufferBytes {
    ResourceIdentity const *identity = nullptr;
    ByteSpan host;
    std::size_t size = 0;
};

template <class T, Dim D>
BufferBytes bufferBytes(BufferResource<T, D> const &buffer) noexcept {
    auto const bytes = buffer.bytes();
    return BufferBytes{&buffer.identity(), ByteSpan{const_cast<std::byte *>(bytes.data()), bytes.size()}, bytes.size()};
}

template <class T, Dim D>
BufferBytes bufferBytes(DeviceBuffer<T, D> const &buffer) noexcept {
    return bufferBytes(*buffer.hostView());
}

template <class B>
struct IsBufferLike : std::false_type {};
template <class T, Dim D>
struct IsBufferLike<BufferResource<T, D>> : std::true_type {};
template <class T, Dim D>
struct IsBufferLike<DeviceBuffer<T, D>> : std::true_type {};

} // namespace detail

} // namespace tc
