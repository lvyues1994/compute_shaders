#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace tc {

enum class Channel : std::uint8_t {
    R = 0,
    G = 1,
    B = 2,
    A = 3,
};

// How the stored bits of a channel are interpreted.
enum class Encoding : std::uint8_t {
    Normalized, // unsigned integer read as [0, 1] float (UNORM)
    Float,      // IEEE float stored as is
    Integer,    // signed or unsigned integer stored as is
};

// Value a missing channel reads as: (0, 0, 0, 1) like GLSL image loads.
template <Encoding E, class T>
constexpr T channelDefault(Channel const c) noexcept {
    if (c != Channel::A) {
        return T{0};
    }
    if constexpr (E == Encoding::Normalized) {
        return std::numeric_limits<T>::max();
    } else {
        return T{1};
    }
}

// Converts one channel between encodings. Integer data never silently turns
// into normalized or float data: those combinations are left undefined.
template <Encoding SrcE, class Src, Encoding DstE, class Dst>
struct ChannelConverter;

template <class Src, class Dst>
struct ChannelConverter<Encoding::Integer, Src, Encoding::Integer, Dst> {
    static constexpr Dst apply(Src const v) noexcept { return static_cast<Dst>(v); }
};

template <class Src, class Dst>
struct ChannelConverter<Encoding::Float, Src, Encoding::Float, Dst> {
    static constexpr Dst apply(Src const v) noexcept { return static_cast<Dst>(v); }
};

template <class Src, class Dst>
struct ChannelConverter<Encoding::Normalized, Src, Encoding::Float, Dst> {
    static constexpr Dst apply(Src const v) noexcept {
        return static_cast<Dst>(v) / static_cast<Dst>(std::numeric_limits<Src>::max());
    }
};

template <class Src, class Dst>
struct ChannelConverter<Encoding::Float, Src, Encoding::Normalized, Dst> {
    static Dst apply(Src const v) noexcept {
        auto const clamped = v < Src{0} ? Src{0} : (v > Src{1} ? Src{1} : v);
        auto const scaled = static_cast<double>(clamped) * std::numeric_limits<Dst>::max();
        return static_cast<Dst>(std::lround(scaled));
    }
};

template <class Src, class Dst>
struct ChannelConverter<Encoding::Normalized, Src, Encoding::Normalized, Dst> {
    static Dst apply(Src const v) noexcept {
        auto const asFloat = ChannelConverter<Encoding::Normalized, Src, Encoding::Float, double>::apply(v);
        return ChannelConverter<Encoding::Float, double, Encoding::Normalized, Dst>::apply(asFloat);
    }
};

namespace cpu {

// A host pixel ("external format"): the listed channels stored in order as T.
// Channels not listed read as channelDefault and ignore writes.
template <Encoding E, class T, Channel... Cs>
struct Pixel {
    static_assert(std::is_arithmetic_v<T> and not std::is_same_v<T, bool>);
    static_assert(E != Encoding::Normalized or std::is_unsigned_v<T>, "normalized channels are unsigned integers");
    static_assert(E != Encoding::Float or std::is_floating_point_v<T>);
    static_assert(E != Encoding::Integer or std::is_integral_v<T>);

    static constexpr Encoding encoding = E;
    static constexpr std::size_t channelCount = sizeof...(Cs);
    static constexpr std::array<Channel, sizeof...(Cs)> layout{Cs...};

    template <Channel C>
    using ChannelType = T;

    T channels[sizeof...(Cs)]{};

    template <Channel C>
    static constexpr int indexOf() noexcept {
        for (std::size_t i = 0; i < channelCount; ++i) {
            if (layout[i] == C) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    template <Channel C>
    constexpr T get() const noexcept {
        if constexpr (indexOf<C>() >= 0) {
            return channels[indexOf<C>()];
        } else {
            return channelDefault<E, T>(C);
        }
    }

    template <Channel C>
    constexpr void set(T const value) noexcept {
        if constexpr (indexOf<C>() >= 0) {
            channels[indexOf<C>()] = value;
        }
    }
};

using R8 = Pixel<Encoding::Normalized, std::uint8_t, Channel::R>;
using RG8 = Pixel<Encoding::Normalized, std::uint8_t, Channel::R, Channel::G>;
using RGB8 = Pixel<Encoding::Normalized, std::uint8_t, Channel::R, Channel::G, Channel::B>;
using RGBA8 = Pixel<Encoding::Normalized, std::uint8_t, Channel::R, Channel::G, Channel::B, Channel::A>;
using BGRA8 = Pixel<Encoding::Normalized, std::uint8_t, Channel::B, Channel::G, Channel::R, Channel::A>;
using R8UI = Pixel<Encoding::Integer, std::uint8_t, Channel::R>;
using RGBA8UI = Pixel<Encoding::Integer, std::uint8_t, Channel::R, Channel::G, Channel::B, Channel::A>;
using RGBA8I = Pixel<Encoding::Integer, std::int8_t, Channel::R, Channel::G, Channel::B, Channel::A>;
using R32UI = Pixel<Encoding::Integer, std::uint32_t, Channel::R>;
using RGBA32UI = Pixel<Encoding::Integer, std::uint32_t, Channel::R, Channel::G, Channel::B, Channel::A>;
using R32I = Pixel<Encoding::Integer, std::int32_t, Channel::R>;
using RGBA32I = Pixel<Encoding::Integer, std::int32_t, Channel::R, Channel::G, Channel::B, Channel::A>;
using R32F = Pixel<Encoding::Float, float, Channel::R>;
using RGB32F = Pixel<Encoding::Float, float, Channel::R, Channel::G, Channel::B>;
using RGBA32F = Pixel<Encoding::Float, float, Channel::R, Channel::G, Channel::B, Channel::A>;

} // namespace cpu

namespace detail {

template <class P, Channel C, class = void>
struct HasChannelAccess : std::false_type {};

template <class P, Channel C>
struct HasChannelAccess<
    P, C,
    std::void_t<typename P::template ChannelType<C>,
                decltype(std::declval<P const &>().template get<C>()),
                decltype(std::declval<P &>().template set<C>(std::declval<P const &>().template get<C>()))>>
    : std::is_convertible<decltype(std::declval<P const &>().template get<C>()),
                          typename P::template ChannelType<C>> {};

} // namespace detail

// C++17 spelling of the talk's PixelConcept: every RGBA channel can be read,
// converted to its channel type, and written back.
template <class P>
inline constexpr bool isPixel = detail::HasChannelAccess<P, Channel::R>::value
    and detail::HasChannelAccess<P, Channel::G>::value and detail::HasChannelAccess<P, Channel::B>::value
    and detail::HasChannelAccess<P, Channel::A>::value;

} // namespace tc
