#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
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

// Host-side storage for a buffer or image: an n-dimensional array of T. A copy
// is a new resource with its own identity; a move transfers the identity.
template <class T, Dim D = Dim::D1>
struct BufferResource {
    static_assert(std::is_trivially_copyable_v<T>, "GPU resources hold trivially copyable elements");
    static_assert(not std::is_same_v<T, bool>, "bool has no GPU memory layout; use uint or int");

    using Traits = DimTraits<D>;
    using IndexType = typename Traits::IndexType;
    using value_type = T;
    static constexpr Dim dim = D;

    explicit BufferResource(IndexType const extent_)
        : extentValue(extent_), elements(static_cast<std::size_t>(Traits::product(extent_))),
          id(ResourceIdentity::create()) {}

    BufferResource(BufferResource const &other)
        : extentValue(other.extentValue), elements(other.elements), id(ResourceIdentity::create()) {}

    BufferResource &operator=(BufferResource const &other) {
        extentValue = other.extentValue;
        elements = other.elements;
        return *this;
    }

    BufferResource(BufferResource &&) noexcept = default;
    BufferResource &operator=(BufferResource &&) noexcept = default;
    ~BufferResource() = default;

    T &operator[](IndexType const coord) noexcept {
        assert(Traits::contains(coord, extentValue));
        return elements[static_cast<std::size_t>(Traits::flatten(coord, extentValue))];
    }

    T const &operator[](IndexType const coord) const noexcept {
        assert(Traits::contains(coord, extentValue));
        return elements[static_cast<std::size_t>(Traits::flatten(coord, extentValue))];
    }

    IndexType extent() const noexcept { return extentValue; }
    std::size_t size() const noexcept { return elements.size(); }
    T *data() noexcept { return elements.data(); }
    T const *data() const noexcept { return elements.data(); }
    auto begin() noexcept { return elements.begin(); }
    auto end() noexcept { return elements.end(); }
    auto begin() const noexcept { return elements.begin(); }
    auto end() const noexcept { return elements.end(); }

    ByteSpan bytes() noexcept {
        return ByteSpan{reinterpret_cast<std::byte *>(elements.data()), elements.size() * sizeof(T)};
    }
    ConstByteSpan bytes() const noexcept {
        return ConstByteSpan{reinterpret_cast<std::byte const *>(elements.data()), elements.size() * sizeof(T)};
    }

    ResourceIdentity const &identity() const noexcept { return id; }

private:
    IndexType extentValue;
    std::vector<T> elements;
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
