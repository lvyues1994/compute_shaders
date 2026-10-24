#pragma once

#include <type_traits>

#include <tc/workgroup.hpp>

namespace tc {

// GLSL `shared T name;`: one instance per workgroup, read and written by all
// of its invocations. Contents are undefined when a workgroup starts on the
// GPU (zero on the CPU). Use `.value` for scalars and `name[i]` for arrays;
// synchronise with barrier().
template <class T>
struct Shared {
    static_assert(std::is_trivially_copyable_v<T>, "shared variables hold plain data");

    using value_type = T;

    T value{};

    template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
    constexpr decltype(auto) operator[](I const index) noexcept {
        return value[index];
    }

    template <class I, std::enable_if_t<std::is_integral_v<I>, int> = 0>
    constexpr decltype(auto) operator[](I const index) const noexcept {
        return value[index];
    }

    constexpr operator T &() noexcept { return value; }
    constexpr operator T const &() const noexcept { return value; }

    constexpr Shared &operator=(T const &value_) noexcept {
        value = value_;
        return *this;
    }
};

} // namespace tc
