#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <tc/dim.hpp>
#include <tc/span.hpp>

namespace tc {

// Identifies one host resource to the GPU backends, which keep a device-side
// mirror per identity. Backends hold only weak references, so destroying the
// resource lets them release the mirror lazily.
struct ResourceIdentity {
    static ResourceIdentity create() { return ResourceIdentity{std::make_shared<Tag>()}; }

    void const *key() const noexcept { return tag.get(); }
    std::weak_ptr<void const> watch() const noexcept { return tag; }
    bool isValid() const noexcept { return tag != nullptr; }

private:
    struct Tag {};
    explicit ResourceIdentity(std::shared_ptr<Tag> tag_) noexcept : tag(std::move(tag_)) {}

    std::shared_ptr<Tag> tag;
};

// Memory that is not on the host heap but still CPU-visible: a mapped GPU
// allocation or an imported platform buffer. `owner` keeps it alive.
struct ExternalMemory {
    std::byte *data = nullptr;
    std::size_t size = 0;
    std::shared_ptr<void> owner;
};

// Storage for a buffer or image: an n-dimensional array of T, on the host
// heap or in external memory. A copy is a new host resource with its own
// identity; a move transfers storage and identity.
template <class T, Dim D = Dim::D1>
struct BufferResource {
    static_assert(std::is_trivially_copyable_v<T>, "GPU resources hold trivially copyable elements");
    static_assert(not std::is_same_v<T, bool>, "bool has no GPU memory layout; use uint or int");

    using Traits = DimTraits<D>;
    using IndexType = typename Traits::IndexType;
    using value_type = T;
    static constexpr Dim dim = D;

    explicit BufferResource(IndexType const extent_) : BufferResource(extent_, ResourceIdentity::create()) {}

    // Elements live in `memory`; used by Device::createBuffer and platform
    // imports. `identity` lets the device recognise its own allocation.
    static BufferResource wrap(IndexType const extent_, ExternalMemory memory, ResourceIdentity identity) {
        auto const count = static_cast<std::size_t>(Traits::product(extent_));
        assert(memory.size >= count * sizeof(T));
        assert(reinterpret_cast<std::uintptr_t>(memory.data) % alignof(T) == 0);
        auto resource = BufferResource{extent_, std::move(identity), Adopt{}};
        resource.first = reinterpret_cast<T *>(memory.data);
        resource.count = count;
        resource.storage = std::move(memory.owner);
        resource.external = true;
        return resource;
    }

    BufferResource(BufferResource const &other) : BufferResource(other.extentValue, ResourceIdentity::create()) {
        std::copy(other.begin(), other.end(), first);
    }

    BufferResource &operator=(BufferResource const &other) {
        if (this != &other) {
            if (count != other.count) {
                auto copy = BufferResource{other};
                copy.id = std::move(id);
                *this = std::move(copy);
            } else {
                extentValue = other.extentValue;
                std::copy(other.begin(), other.end(), first);
            }
        }
        return *this;
    }

    BufferResource(BufferResource &&other) noexcept
        : extentValue(other.extentValue), storage(std::move(other.storage)),
          first(std::exchange(other.first, nullptr)), count(std::exchange(other.count, 0)),
          external(other.external), id(std::move(other.id)) {}

    BufferResource &operator=(BufferResource &&other) noexcept {
        extentValue = other.extentValue;
        storage = std::move(other.storage);
        first = std::exchange(other.first, nullptr);
        count = std::exchange(other.count, 0);
        external = other.external;
        id = std::move(other.id);
        return *this;
    }

    ~BufferResource() = default;

    T &operator[](IndexType const coord) noexcept {
        assert(Traits::contains(coord, extentValue));
        return first[static_cast<std::size_t>(Traits::flatten(coord, extentValue))];
    }

    T const &operator[](IndexType const coord) const noexcept {
        assert(Traits::contains(coord, extentValue));
        return first[static_cast<std::size_t>(Traits::flatten(coord, extentValue))];
    }

    IndexType extent() const noexcept { return extentValue; }
    std::size_t size() const noexcept { return count; }
    T *data() noexcept { return first; }
    T const *data() const noexcept { return first; }
    T *begin() noexcept { return first; }
    T *end() noexcept { return first + count; }
    T const *begin() const noexcept { return first; }
    T const *end() const noexcept { return first + count; }

    // True when the elements live in device-mapped or imported memory.
    bool usesExternalMemory() const noexcept { return external; }

    ByteSpan bytes() noexcept { return ByteSpan{reinterpret_cast<std::byte *>(first), count * sizeof(T)}; }
    ConstByteSpan bytes() const noexcept {
        return ConstByteSpan{reinterpret_cast<std::byte const *>(first), count * sizeof(T)};
    }

    ResourceIdentity const &identity() const noexcept { return id; }

private:
    struct Adopt {};

    BufferResource(IndexType const extent_, ResourceIdentity identity, Adopt) noexcept
        : extentValue(extent_), id(std::move(identity)) {}

    BufferResource(IndexType const extent_, ResourceIdentity identity)
        : extentValue(extent_), count(static_cast<std::size_t>(Traits::product(extent_))), id(std::move(identity)) {
        auto elements = std::make_shared<std::vector<T>>(count);
        first = elements->data();
        storage = std::move(elements);
    }

    IndexType extentValue;
    std::shared_ptr<void> storage;
    T *first = nullptr;
    std::size_t count = 0;
    bool external = false;
    ResourceIdentity id;
};

// A shader storage buffer binding: `layout(std430, binding = Binding) buffer
// { T name[]; }`. On the CPU it forwards element access to the attached
// resource; on the GPU the backend binds the resource's device mirror.
template <class T, std::uint32_t Binding>
struct BufferBinding {
    using value_type = T;
    using Resource = BufferResource<T, Dim::D1>;
    static constexpr std::uint32_t binding = Binding;

    void attach(Resource &resource_) noexcept { resource = &resource_; }
    Resource *attachedResource() const noexcept { return resource; }

    template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
    T &operator[](I const index) const noexcept {
        assert(resource != nullptr and "BufferBinding used before attach()");
        return (*resource)[static_cast<std::int32_t>(index)];
    }

    // GLSL `name.length()` on the runtime-sized array; returns int like GLSL.
    std::int32_t length() const noexcept {
        assert(resource != nullptr and "BufferBinding used before attach()");
        return static_cast<std::int32_t>(resource->size());
    }

private:
    Resource *resource = nullptr;
};

} // namespace tc
