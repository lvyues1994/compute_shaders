#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <type_traits>

#include <tc/pixel.hpp>
#include <tc/vec.hpp>

namespace tc {

namespace gpu {

// A GPU image format ("internal format"): ChannelCount channels (R, RG or
// RGBA) of type T with the given encoding, e.g. rgba8 or r32ui.
template <Encoding E, class T, int ChannelCount>
struct Format {
    static_assert(ChannelCount == 1 or ChannelCount == 2 or ChannelCount == 4,
                  "GLSL image formats have 1, 2 or 4 channels");
    static constexpr Encoding encoding = E;
    using StorageType = T;
    static constexpr int channelCount = ChannelCount;
};

using R8 = Format<Encoding::Normalized, std::uint8_t, 1>;
using RG8 = Format<Encoding::Normalized, std::uint8_t, 2>;
using RGBA8 = Format<Encoding::Normalized, std::uint8_t, 4>;
using R8UI = Format<Encoding::Integer, std::uint8_t, 1>;
using RGBA8UI = Format<Encoding::Integer, std::uint8_t, 4>;
using RGBA8I = Format<Encoding::Integer, std::int8_t, 4>;
using R32UI = Format<Encoding::Integer, std::uint32_t, 1>;
using RG32UI = Format<Encoding::Integer, std::uint32_t, 2>;
using RGBA32UI = Format<Encoding::Integer, std::uint32_t, 4>;
using R32I = Format<Encoding::Integer, std::int32_t, 1>;
using RGBA32I = Format<Encoding::Integer, std::int32_t, 4>;
using R32F = Format<Encoding::Float, float, 1>;
using RG32F = Format<Encoding::Float, float, 2>;
using RGBA32F = Format<Encoding::Float, float, 4>;
using R16F = Format<Encoding::Float, half, 1>;
using RG16F = Format<Encoding::Float, half, 2>;
using RGBA16F = Format<Encoding::Float, half, 4>; // storage images on every target; filterable on OpenGL ES

} // namespace gpu

// Runtime description of a GPU image format, passed to the backends.
struct ImageFormat {
    Encoding encoding = Encoding::Normalized;
    bool isSigned = false;
    std::uint8_t bitsPerChannel = 8;
    std::uint8_t channelCount = 4;

    std::size_t bytesPerTexel() const noexcept { return std::size_t{bitsPerChannel} / 8 * channelCount; }
};

inline bool operator==(ImageFormat const &a, ImageFormat const &b) noexcept {
    return a.encoding == b.encoding and a.isSigned == b.isSigned and a.bitsPerChannel == b.bitsPerChannel
        and a.channelCount == b.channelCount;
}

// GLSL layout qualifier of a format, e.g. "rgba8", "r32ui", "rg32f".
inline std::string glslFormatQualifier(ImageFormat const &format) {
    static constexpr std::array<char const *, 5> channelPrefix{"", "r", "rg", "rgb", "rgba"};
    auto qualifier = std::string{channelPrefix[format.channelCount]} + std::to_string(format.bitsPerChannel);
    switch (format.encoding) {
    case Encoding::Normalized:
        return format.isSigned ? qualifier + "_snorm" : qualifier;
    case Encoding::Float:
        return qualifier + "f";
    case Encoding::Integer:
        return qualifier + (format.isSigned ? "i" : "ui");
    }
    return qualifier;
}

// Formats usable as storage images on every target: OpenGL ES 3.1 and the
// formats Vulkan requires all devices to support for storage images.
inline bool isPortableStorageFormat(ImageFormat const &format) {
    auto const q = glslFormatQualifier(format);
    return q == "rgba8" or q == "rgba8ui" or q == "rgba8i" or q == "r32ui" or q == "r32i" or q == "r32f"
        or q == "rgba32ui" or q == "rgba32i" or q == "rgba32f" or q == "rgba16f";
}

namespace detail {

template <Encoding E, class T, int N>
struct StoragePixelFor;

template <Encoding E, class T>
struct StoragePixelFor<E, T, 1> {
    using type = cpu::Pixel<E, T, Channel::R>;
};

template <Encoding E, class T>
struct StoragePixelFor<E, T, 2> {
    using type = cpu::Pixel<E, T, Channel::R, Channel::G>;
};

template <Encoding E, class T>
struct StoragePixelFor<E, T, 4> {
    using type = cpu::Pixel<E, T, Channel::R, Channel::G, Channel::B, Channel::A>;
};

template <Encoding E, class T>
struct ShaderChannel {
    using type = float;
};

template <class T>
struct ShaderChannel<Encoding::Integer, T> {
    using type = std::conditional_t<std::is_signed_v<T>, std::int32_t, std::uint32_t>;
};

} // namespace detail

// What a GLSL shader sees of an image format: imageLoad returns VectorType
// (vec4, ivec4 or uvec4); StoragePixel is the exact texel layout in memory.
template <class G>
struct GPUFormatTraits {
    using Format = G;
    using StorageType = typename G::StorageType;
    using StoragePixel = typename detail::StoragePixelFor<G::encoding, StorageType, G::channelCount>::type;
    using ChannelType = typename detail::ShaderChannel<G::encoding, StorageType>::type;
    using VectorType = vec<ChannelType, 4>;
    static constexpr Encoding shaderEncoding = G::encoding == Encoding::Integer ? Encoding::Integer : Encoding::Float;

    static constexpr ImageFormat format{G::encoding, std::is_signed_v<StorageType>,
                                        static_cast<std::uint8_t>(8 * sizeof(StorageType)),
                                        static_cast<std::uint8_t>(G::channelCount)};
};

// Pixels convert between host and GPU formats only within the same family:
// integer data stays integer, normalized/float data stays non-integer.
template <class G, class P>
inline constexpr bool isUploadCompatible = (G::encoding == Encoding::Integer) == (P::encoding == Encoding::Integer);

namespace detail {

template <class DstPixel, class SrcPixel, Channel C>
constexpr void convertChannel(DstPixel &dst, SrcPixel const &src) noexcept {
    using Src = typename SrcPixel::template ChannelType<C>;
    using Dst = typename DstPixel::template ChannelType<C>;
    dst.template set<C>(ChannelConverter<SrcPixel::encoding, Src, DstPixel::encoding, Dst>::apply(src.template get<C>()));
}

} // namespace detail

// Converts a pixel between two formats channel by channel.
template <class DstPixel, class SrcPixel>
DstPixel convertPixel(SrcPixel const &src) noexcept {
    if constexpr (std::is_same_v<DstPixel, SrcPixel>) {
        return src;
    } else {
        auto dst = DstPixel{};
        detail::convertChannel<DstPixel, SrcPixel, Channel::R>(dst, src);
        detail::convertChannel<DstPixel, SrcPixel, Channel::G>(dst, src);
        detail::convertChannel<DstPixel, SrcPixel, Channel::B>(dst, src);
        detail::convertChannel<DstPixel, SrcPixel, Channel::A>(dst, src);
        return dst;
    }
}

// The value a GLSL imageLoad returns for a stored texel.
template <class G>
typename GPUFormatTraits<G>::VectorType decodeTexel(typename GPUFormatTraits<G>::StoragePixel const &texel) noexcept {
    using Traits = GPUFormatTraits<G>;
    using Storage = typename Traits::StorageType;
    using Shader = typename Traits::ChannelType;
    using Convert = ChannelConverter<G::encoding, Storage, Traits::shaderEncoding, Shader>;
    auto const read = [&](auto const channel) {
        constexpr auto c = decltype(channel)::value;
        return Convert::apply(texel.template get<c>());
    };
    return typename Traits::VectorType{read(std::integral_constant<Channel, Channel::R>{}),
                                       read(std::integral_constant<Channel, Channel::G>{}),
                                       read(std::integral_constant<Channel, Channel::B>{}),
                                       read(std::integral_constant<Channel, Channel::A>{})};
}

// The texel a GLSL imageStore writes for a shader value.
template <class G>
typename GPUFormatTraits<G>::StoragePixel encodeTexel(typename GPUFormatTraits<G>::VectorType const &value) noexcept {
    using Traits = GPUFormatTraits<G>;
    using Storage = typename Traits::StorageType;
    using Shader = typename Traits::ChannelType;
    using Convert = ChannelConverter<Traits::shaderEncoding, Shader, G::encoding, Storage>;
    auto texel = typename Traits::StoragePixel{};
    texel.template set<Channel::R>(Convert::apply(value.x));
    texel.template set<Channel::G>(Convert::apply(value.y));
    texel.template set<Channel::B>(Convert::apply(value.z));
    texel.template set<Channel::A>(Convert::apply(value.w));
    return texel;
}

} // namespace tc
