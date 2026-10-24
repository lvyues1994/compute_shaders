#pragma once

#include <cstdint>
#include <type_traits>

#include <tc/vec.hpp>

namespace tc {

// A value the host sets per dispatch: `layout(location = Location) uniform T`
// on OpenGL, a push constant on Vulkan. Kernels read it like a plain T.
template <class T, std::uint32_t Location>
struct Uniform {
    static_assert(not std::is_same_v<T, bool> and not std::is_same_v<detail::ComponentOf<T>, bool>,
                  "bool uniforms have no portable layout; use int or uint");

    using value_type = T;
    static constexpr std::uint32_t location = Location;

    T value{};

    constexpr operator T const &() const noexcept { return value; }

    constexpr Uniform &operator=(T const &value_) noexcept {
        value = value_;
        return *this;
    }
};

namespace detail {

template <class T, std::uint32_t L>
struct ComponentSource<Uniform<T, L>> {
    static constexpr int count = ComponentSourceOf<T>::count;

    template <class V>
    static constexpr void append(V &target, int &cursor, Uniform<T, L> const &uniform) noexcept {
        ComponentSourceOf<T>::append(target, cursor, uniform.value);
    }
};

} // namespace detail

} // namespace tc
