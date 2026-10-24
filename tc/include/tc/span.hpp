#pragma once

#include <cstddef>
#include <type_traits>

namespace tc {

// Non-owning view over contiguous elements (std::span is C++20).
template <class T>
struct Span {
    constexpr Span() noexcept = default;
    constexpr Span(T *const data_, std::size_t const size_) noexcept : first(data_), count(size_) {}

    template <class U, std::enable_if_t<std::is_convertible_v<U (*)[], T (*)[]>, int> = 0>
    constexpr Span(Span<U> const &other) noexcept : first(other.data()), count(other.size()) {}

    constexpr T *data() const noexcept { return first; }
    constexpr std::size_t size() const noexcept { return count; }
    constexpr bool empty() const noexcept { return count == 0; }
    constexpr T *begin() const noexcept { return first; }
    constexpr T *end() const noexcept { return first + count; }
    constexpr T &operator[](std::size_t const i) const noexcept { return first[i]; }

private:
    T *first = nullptr;
    std::size_t count = 0;
};

using ByteSpan = Span<std::byte>;
using ConstByteSpan = Span<std::byte const>;

} // namespace tc
