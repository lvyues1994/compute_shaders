#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <type_traits>

#include <tc/buffer.hpp>
#include <tc/dim.hpp>
#include <tc/gpu_format.hpp>
#include <tc/mip_chain.hpp>
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

// How textureLod chooses between mip levels: None samples level 0 only,
// Nearest the closest level, Linear blends the two closest.
enum class Mip : std::uint8_t {
    None,
    Nearest,
    Linear,
};

// A read-only sampled texture: `layout(binding = Binding) uniform sampler<D>`.
// Unlike storage images, any sampleable format works, including r8 and rg8
// on OpenGL ES (the Y and UV planes of NV12). Filter, Wrap and Mip form the
// sampler state used by texture() and textureLod(); texelFetch() ignores
// them. Attach a BufferResource (one level) or a MipChain.
template <class G, Dim D, class P, std::uint32_t Binding, Filter F = Filter::Nearest, Wrap W = Wrap::ClampToEdge,
          Mip M = Mip::None>
struct TextureBinding {
    static_assert(D != Dim::D1, "1D textures are unavailable on OpenGL ES; use a BufferBinding instead");
    static_assert(isPixel<P>, "P must model the pixel concept (see tc/pixel.hpp)");
    static_assert(isUploadCompatible<G, P>, "integer pixels need an integer GPU format and vice versa");
    static_assert((F == Filter::Nearest and M != Mip::Linear) or G::encoding != Encoding::Integer,
                  "integer textures cannot be filtered");

    using Format = G;
    using Pixel = P;
    using Traits = GPUFormatTraits<G>;
    using Resource = BufferResource<P, D>;
    using Chain = MipChain<P, D>;
    using IndexType = typename DimTraits<D>::IndexType;
    static constexpr Dim dim = D;
    static constexpr std::uint32_t binding = Binding;
    static constexpr Filter filter = F;
    static constexpr Wrap wrap = W;
    static constexpr Mip mip = M;

    void attach(Resource &resource_) noexcept { target = detail::ImageAttachment<P, D>{&resource_, nullptr, 0}; }
    void attach(Chain &chain_) noexcept { target = detail::ImageAttachment<P, D>{nullptr, &chain_, 0}; }

    // Level 0.
    Resource *attachedResource() const noexcept { return target.resource(0); }
    std::uint32_t levelCount() const noexcept { return target.levels(); }
    Resource *levelResource(std::uint32_t const index) const noexcept { return target.resource(index); }
    ResourceIdentity const *identity() const noexcept { return target.identity(); }

private:
    detail::ImageAttachment<P, D> target;
};

template <class T>
struct IsTextureBinding : std::false_type {};

template <class G, Dim D, class P, std::uint32_t B, Filter F, Wrap W, Mip M>
struct IsTextureBinding<TextureBinding<G, D, P, B, F, W, M>> : std::true_type {};

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
template <>
struct TexCoord<Dim::D2Array> {
    using type = vec3; // u, v and an unnormalized layer
};

// Axes that are filtered and wrapped: x and y, and z of 3D textures.
template <Dim D>
inline constexpr int filteredAxes = D == Dim::D3 ? 3 : 2;

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
typename T::Traits::VectorType texelAt(typename T::Resource const &level, typename T::IndexType const coord) noexcept {
    using Traits = typename T::Traits;
    auto const stored = convertPixel<typename Traits::StoragePixel>(level[coord]);
    return decodeTexel<typename T::Format>(stored);
}

template <class T>
typename T::IndexType wrapped(typename T::IndexType coord, typename T::IndexType const size) noexcept {
    for (int i = 0; i < filteredAxes<T::dim>; ++i) {
        coord[i] = wrapCoordinate(coord[i], size[i], T::wrap);
    }
    return coord;
}

// The layer texture() reads from an unnormalized layer coordinate.
inline std::int32_t arrayLayer(float const layer, std::int32_t const layers) noexcept {
    auto const nearest = static_cast<std::int32_t>(std::floor(layer + 0.5f));
    return nearest < 0 ? 0 : (nearest >= layers ? layers - 1 : nearest);
}

// One level, filtered with the binding's Filter (OpenGL's nearest and
// bilinear rules).
template <class T>
typename T::Traits::VectorType sampleLevel(typename T::Resource const &level,
                                           typename TexCoord<T::dim>::type const &uv) noexcept {
    using Vector = typename T::Traits::VectorType;
    using Index = typename T::IndexType;
    constexpr int n = filteredAxes<T::dim>;
    auto const size = level.extent();
    auto base = Index{};
    if constexpr (T::dim == Dim::D2Array) {
        base[2] = arrayLayer(uv[2], size[2]);
    }
    if constexpr (T::filter == Filter::Nearest) {
        for (int i = 0; i < n; ++i) {
            base[i] = static_cast<std::int32_t>(std::floor(uv[i] * static_cast<float>(size[i])));
        }
        return texelAt<T>(level, wrapped<T>(base, size));
    } else {
        auto weight = vec3{};
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
            result += texelAt<T>(level, wrapped<T>(coord, size)) * w;
        }
        return result;
    }
}

template <class T>
typename T::Resource const &requireLevel(T const &texture, std::int32_t const lod) noexcept {
    assert(texture.attachedResource() != nullptr and "TextureBinding used before attach()");
    assert(lod >= 0 and static_cast<std::uint32_t>(lod) < texture.levelCount() and "mip level out of range");
    return *texture.levelResource(static_cast<std::uint32_t>(lod));
}

} // namespace detail

// GLSL textureSize of mip level `lod` (array textures: layers in z).
template <class G, Dim D, class P, std::uint32_t B, Filter F, Wrap W, Mip M>
typename DimTraits<D>::IndexType textureSize(TextureBinding<G, D, P, B, F, W, M> const &texture,
                                             std::int32_t const lod) noexcept {
    return detail::requireLevel(texture, lod).extent();
}

// GLSL texelFetch from mip level `lod`. Out-of-range fetches return zero.
template <class G, Dim D, class P, std::uint32_t B, Filter F, Wrap W, Mip M>
typename GPUFormatTraits<G>::VectorType texelFetch(TextureBinding<G, D, P, B, F, W, M> const &texture,
                                                   typename DimTraits<D>::IndexType const coord,
                                                   std::int32_t const lod) noexcept {
    using Texture = TextureBinding<G, D, P, B, F, W, M>;
    auto const &level = detail::requireLevel(texture, lod);
    if (not DimTraits<D>::contains(coord, level.extent())) {
        return typename GPUFormatTraits<G>::VectorType{};
    }
    return detail::texelAt<Texture>(level, coord);
}

// GLSL textureLod: normalized coordinates (array textures: an unnormalized
// layer in the last component), sampled at level of detail `lod` with the
// binding's sampler state, by the OpenGL / Vulkan rules: lod <= 0 samples
// level 0; Mip::Nearest picks level ceil(lod + 0.5) - 1, Mip::Linear blends
// floor(lod) and the next level; both stop at the last level.
template <class G, Dim D, class P, std::uint32_t B, Filter F, Wrap W, Mip M>
typename GPUFormatTraits<G>::VectorType textureLod(TextureBinding<G, D, P, B, F, W, M> const &texture,
                                                   typename detail::TexCoord<D>::type const &uv,
                                                   float const lod) noexcept {
    using Texture = TextureBinding<G, D, P, B, F, W, M>;
    assert(texture.attachedResource() != nullptr and "TextureBinding used before attach()");
    auto const last = static_cast<float>(texture.levelCount() - 1);
    auto const at = [&](float const level) {
        return detail::sampleLevel<Texture>(*texture.levelResource(static_cast<std::uint32_t>(level)), uv);
    };
    if constexpr (M == Mip::None) {
        (void)last;
        (void)lod;
        return at(0.0f);
    } else if constexpr (M == Mip::Nearest) {
        auto const level = lod <= 0.5f ? 0.0f : std::ceil(lod + 0.5f) - 1.0f;
        return at(std::fmin(level, last));
    } else {
        if (not(lod > 0.0f)) {
            return at(0.0f);
        }
        auto const low = std::fmin(std::floor(lod), last);
        auto const high = std::fmin(low + 1.0f, last);
        auto const weight = lod - std::floor(lod);
        return low == high ? at(low) : at(low) * (1.0f - weight) + at(high) * weight;
    }
}

// GLSL texture(): compute shaders have no derivatives, so the level of
// detail is 0, as on the GPU.
template <class G, Dim D, class P, std::uint32_t B, Filter F, Wrap W, Mip M>
typename GPUFormatTraits<G>::VectorType texture(TextureBinding<G, D, P, B, F, W, M> const &texture,
                                                typename detail::TexCoord<D>::type const &uv) noexcept {
    return textureLod(texture, uv, 0.0f);
}

} // namespace tc
