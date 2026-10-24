#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include <tc/buffer.hpp>
#include <tc/dim.hpp>
#include <tc/gpu_format.hpp>
#include <tc/pixel.hpp>

namespace tc {

enum class Filter : std::uint8_t {
    Nearest,
    Linear,
};

enum class Wrap : std::uint8_t {
    ClampToEdge,
    Repeat,
    MirroredRepeat,
};

// A read-only sampled texture: `layout(binding = Binding) uniform sampler<D>`.
// Unlike storage images, any sampleable format works, including r8 and rg8
// on OpenGL ES (the Y and UV planes of NV12). Filter and Wrap form the
// sampler state used by texture(); texelFetch() ignores them.
template <class G, Dim D, class P, std::uint32_t Binding, Filter F = Filter::Nearest, Wrap W = Wrap::ClampToEdge>
struct TextureBinding {
    static_assert(D != Dim::D1, "1D textures are unavailable on OpenGL ES; use a BufferBinding instead");
    static_assert(isPixel<P>, "P must model the pixel concept (see tc/pixel.hpp)");
    static_assert(isUploadCompatible<G, P>, "integer pixels need an integer GPU format and vice versa");
    static_assert(F == Filter::Nearest or G::encoding != Encoding::Integer, "integer textures cannot be filtered");

    using Format = G;
    using Pixel = P;
    using Traits = GPUFormatTraits<G>;
    using Resource = BufferResource<P, D>;
    using IndexType = typename DimTraits<D>::IndexType;
    static constexpr Dim dim = D;
    static constexpr std::uint32_t binding = Binding;
    static constexpr Filter filter = F;
    static constexpr Wrap wrap = W;

    void attach(Resource &resource_) noexcept { resource = &resource_; }
    Resource *attachedResource() const noexcept { return resource; }

private:
    Resource *resource = nullptr;
};

template <class T>
struct IsTextureBinding : std::false_type {};

template <class G, Dim D, class P, std::uint32_t B, Filter F, Wrap W>
struct IsTextureBinding<TextureBinding<G, D, P, B, F, W>> : std::true_type {};

namespace detail {

template <Dim D>
struct TexCoord;
template <>
struct TexCoord<Dim::D2> {
    using type = vec2;
};
template <>
struct TexCoord<Dim::D3> {
    using type = vec3;
};

inline std::int32_t wrapCoordinate(std::int32_t const i, std::int32_t const size, Wrap const wrap) noexcept {
    switch (wrap) {
    case Wrap::ClampToEdge:
        return i < 0 ? 0 : (i >= size ? size - 1 : i);
    case Wrap::Repeat: {
        auto const m = i % size;
        return m < 0 ? m + size : m;
    }
    case Wrap::MirroredRepeat: {
        auto const period = 2 * size;
        auto m = i % period;
        m = m < 0 ? m + period : m;
        return m < size ? m : period - 1 - m;
    }
    }
    return i;
}

template <class T>
typename T::Traits::VectorType texelAt(T const &texture, typename T::IndexType const coord) noexcept {
    auto const *resource = texture.attachedResource();
    using Traits = typename T::Traits;
    auto const stored = convertPixel<typename Traits::StoragePixel>((*resource)[coord]);
    return decodeTexel<typename T::Format>(stored);
}

template <class T>
typename T::IndexType wrapped(typename T::IndexType coord, typename T::IndexType const size) noexcept {
    if constexpr (T::dim == Dim::D2) {
        return ivec2{wrapCoordinate(coord.x, size.x, T::wrap), wrapCoordinate(coord.y, size.y, T::wrap)};
    } else {
        return ivec3{wrapCoordinate(coord.x, size.x, T::wrap), wrapCoordinate(coord.y, size.y, T::wrap),
                     wrapCoordinate(coord.z, size.z, T::wrap)};
    }
}

} // namespace detail

// GLSL textureSize at mip level 0.
template <class G, Dim D, class P, std::uint32_t B, Filter F, Wrap W>
typename DimTraits<D>::IndexType textureSize(TextureBinding<G, D, P, B, F, W> const &texture,
                                             std::int32_t const lod) noexcept {
    assert(texture.attachedResource() != nullptr and "TextureBinding used before attach()");
    assert(lod == 0 and "textures have a single mip level");
    (void)lod;
    return texture.attachedResource()->extent();
}

// GLSL texelFetch at mip level 0. Out-of-range fetches return zero.
template <class G, Dim D, class P, std::uint32_t B, Filter F, Wrap W>
typename GPUFormatTraits<G>::VectorType texelFetch(TextureBinding<G, D, P, B, F, W> const &texture,
                                                   typename DimTraits<D>::IndexType const coord,
                                                   std::int32_t const lod) noexcept {
    assert(texture.attachedResource() != nullptr and "TextureBinding used before attach()");
    assert(lod == 0 and "textures have a single mip level");
    (void)lod;
    if (not DimTraits<D>::contains(coord, texture.attachedResource()->extent())) {
        return typename GPUFormatTraits<G>::VectorType{};
    }
    return detail::texelAt(texture, coord);
}

// GLSL texture(): normalized coordinates, filtered and wrapped as the
// binding's sampler state says (OpenGL's nearest / bilinear rules).
template <class G, Dim D, class P, std::uint32_t B, Filter F, Wrap W>
typename GPUFormatTraits<G>::VectorType texture(TextureBinding<G, D, P, B, F, W> const &texture,
                                                typename detail::TexCoord<D>::type const &uv) noexcept {
    using Texture = TextureBinding<G, D, P, B, F, W>;
    using Vector = typename GPUFormatTraits<G>::VectorType;
    assert(texture.attachedResource() != nullptr and "TextureBinding used before attach()");
    auto const size = texture.attachedResource()->extent();
    constexpr int n = D == Dim::D2 ? 2 : 3;
    if constexpr (F == Filter::Nearest) {
        auto coord = typename Texture::IndexType{};
        for (int i = 0; i < n; ++i) {
            coord[i] = static_cast<std::int32_t>(std::floor(uv[i] * static_cast<float>(size[i])));
        }
        return detail::texelAt(texture, detail::wrapped<Texture>(coord, size));
    } else {
        auto base = typename Texture::IndexType{};
        auto weight = typename detail::TexCoord<D>::type{};
        for (int i = 0; i < n; ++i) {
            auto const u = uv[i] * static_cast<float>(size[i]) - 0.5f;
            auto const floored = std::floor(u);
            base[i] = static_cast<std::int32_t>(floored);
            weight[i] = u - floored;
        }
        auto result = Vector{};
        for (int corner = 0; corner < (1 << n); ++corner) {
            auto coord = base;
            auto w = 1.0f;
            for (int i = 0; i < n; ++i) {
                auto const upper = (corner >> i) & 1;
                coord[i] += upper;
                w *= upper != 0 ? weight[i] : 1.0f - weight[i];
            }
            result += detail::texelAt(texture, detail::wrapped<Texture>(coord, size)) * w;
        }
        return result;
    }
}

} // namespace tc
