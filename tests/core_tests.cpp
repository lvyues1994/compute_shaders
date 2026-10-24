#include <cmath>
#include <cstdint>

#include <gtest/gtest.h>

#include <tc/cpu/cpu_device.hpp>
#include <tc/device.hpp>
#include <tc/tc.hpp>

namespace {

using namespace tc;

TEST(Vec, ConstructsLikeGlsl) {
    auto const broadcast = vec3{2.0f};
    EXPECT_EQ(broadcast, (vec3{2.0f, 2.0f, 2.0f}));
    auto const concatenated = vec4{vec2{1.0f, 2.0f}, 3, 4.0};
    EXPECT_EQ(concatenated, (vec4{1.0f, 2.0f, 3.0f, 4.0f}));
    auto const converted = ivec3{vec3{1.9f, -1.9f, 0.5f}};
    EXPECT_EQ(converted, (ivec3{1, -1, 0}));
    auto const truncated = vec2{vec4{5.0f, 6.0f, 7.0f, 8.0f}};
    EXPECT_EQ(truncated, (vec2{5.0f, 6.0f}));
}

TEST(Vec, SwizzlesAndComponents) {
    auto const v = vec4{1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_EQ(v.xy(), (vec2{1.0f, 2.0f}));
    EXPECT_EQ(v.wzyx(), (vec4{4.0f, 3.0f, 2.0f, 1.0f}));
    EXPECT_EQ(v.zzx(), (vec3{3.0f, 3.0f, 1.0f}));
    EXPECT_FLOAT_EQ(v[2], 3.0f);
    EXPECT_EQ(vec4::length(), 4);
}

TEST(Vec, ArithmeticIsComponentWise) {
    auto v = vec3{1.0f, 2.0f, 3.0f};
    EXPECT_EQ(v * 2.0f, (vec3{2.0f, 4.0f, 6.0f}));
    EXPECT_EQ(1.0f - v, (vec3{0.0f, -1.0f, -2.0f}));
    EXPECT_EQ(v * v, (vec3{1.0f, 4.0f, 9.0f}));
    v += vec3{1.0f};
    EXPECT_EQ(v, (vec3{2.0f, 3.0f, 4.0f}));
    EXPECT_EQ((uvec2{6u, 7u} % 4u), (uvec2{2u, 3u}));
    EXPECT_EQ((ivec2{1, 2} << 2), (ivec2{4, 8}));
}

TEST(Vec, MatchesStd430Layout) {
    EXPECT_EQ(sizeof(vec3), 16u);
    EXPECT_EQ(alignof(vec3), 16u);
    EXPECT_EQ(sizeof(dvec2), 16u);
}

TEST(Uniform, ReadsLikeItsValue) {
    auto const pad = Uniform<std::int32_t, 1>{3};
    auto const offset = Uniform<ivec2, 2>{ivec2{1, 2}};
    EXPECT_EQ(pad + 1, 4);
    EXPECT_EQ(ivec2{5} + offset, (ivec2{6, 7}));
    EXPECT_EQ((ivec3{offset, pad}), (ivec3{1, 2, 3}));
}

TEST(GlslMath, FollowsGlslDefinitions) {
    EXPECT_FLOAT_EQ(mod(-1.0f, 3.0f), 2.0f);
    EXPECT_FLOAT_EQ(fract(-1.25f), 0.75f);
    EXPECT_FLOAT_EQ(mix(2.0f, 4.0f, 0.25f), 2.5f);
    EXPECT_FLOAT_EQ(smoothstep(0.0f, 1.0f, 0.5f), 0.5f);
    EXPECT_EQ(clamp(ivec2{-5, 5}, 0, 3), (ivec2{0, 3}));
    EXPECT_FLOAT_EQ(length(vec2{3.0f, 4.0f}), 5.0f);
    EXPECT_EQ(cross(vec3{1.0f, 0.0f, 0.0f}, vec3{0.0f, 1.0f, 0.0f}), (vec3{0.0f, 0.0f, 1.0f}));
    EXPECT_TRUE(all(lessThan(ivec3{1, 2, 3}, ivec3{2, 3, 4})));
    EXPECT_FALSE(any(not_(bvec2{true, true})));
}

TEST(Dim, FlattensRowMajor) {
    using D2 = DimTraits<Dim::D2>;
    using D3 = DimTraits<Dim::D3>;
    EXPECT_EQ(D2::flatten(ivec2{3, 2}, ivec2{10, 5}), 23);
    EXPECT_EQ(D3::flatten(ivec3{1, 2, 3}, ivec3{4, 5, 6}), (3 * 5 + 2) * 4 + 1);
    EXPECT_EQ(D3::product(ivec3{4, 5, 6}), 120);
}

TEST(Pixel, PadsMissingChannels) {
    auto const px = cpu::RGB8{{51, 102, 204}};
    static_assert(isPixel<cpu::RGB8>);
    EXPECT_EQ(px.get<Channel::G>(), 102);
    EXPECT_EQ(px.get<Channel::A>(), 255);
    auto const asFloat = convertPixel<cpu::RGBA32F>(px);
    EXPECT_FLOAT_EQ(asFloat.get<Channel::R>(), 0.2f);
    EXPECT_FLOAT_EQ(asFloat.get<Channel::A>(), 1.0f);
}

TEST(Pixel, ConvertsLikeTheTalk) {
    // 92 stored as an unsigned byte reads as 0.36 in a normalized format.
    auto const value = ChannelConverter<Encoding::Normalized, std::uint8_t, Encoding::Float, float>::apply(92);
    EXPECT_NEAR(value, 0.36f, 0.001f);
    auto const back = ChannelConverter<Encoding::Float, float, Encoding::Normalized, std::uint8_t>::apply(value);
    EXPECT_EQ(back, 92);
}

TEST(Image, LoadsAndStoresThroughTheGpuFormat) {
    auto pixels = BufferResource<cpu::R8UI, Dim::D2>{ivec2{4, 4}};
    pixels[ivec2{1, 2}].channels[0] = 7;
    auto image = ImageBinding<gpu::R32UI, Dim::D2, cpu::R8UI, 0>{};
    image.attach(pixels);
    EXPECT_EQ(imageLoad(image, ivec2{1, 2}), (uvec4{7u, 0u, 0u, 1u}));
    volatile int outside = 9; // keeps GCC from flagging the guarded out-of-bounds path
    EXPECT_EQ(imageLoad(image, ivec2{outside, outside}), (uvec4{0u}));
    auto const origin = ivec2{0, 0};
    imageStore(image, origin, uvec4{300u});
    EXPECT_EQ(pixels[origin].channels[0], static_cast<std::uint8_t>(300u));
    EXPECT_EQ(imageSize(image), (ivec2{4, 4}));
}

TEST(BufferResource, CopiesGetANewIdentity) {
    auto a = BufferResource<float>{8};
    auto const b = a;
    EXPECT_NE(a.identity().key(), b.identity().key());
    auto const key = a.identity().key();
    auto const moved = std::move(a);
    EXPECT_EQ(moved.identity().key(), key);
}

struct TC_KERNEL Doubler {
    static constexpr char fileLocation[] = "doubler";
    uvec3 local_size{8, 1, 1};

    BufferBinding<std::int32_t, 0> data;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i < uint(data.length())) {
            data[i] = data[i] * 2 + std::int32_t(gl_LocalInvocationIndex);
        }
    }
};

TEST(CpuDevice, RunsKernelsWithoutTranspilation) {
    auto device = cpu::createDevice(cpu::CpuDeviceConfig{3});
    auto data = BufferResource<std::int32_t>{100};
    for (int i = 0; i < 100; ++i) {
        data[i] = i;
    }
    auto kernel = Doubler{};
    kernel.data.attach(data);
    device->useKernel(kernel);
    device->bindBuffer(kernel.data);
    device->execute(kernel, uvec3{13u, 1u, 1u});
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(data[i], 2 * i + i % 8);
    }
}

TEST(CpuDevice, RejectsUnattachedBindings) {
    auto device = cpu::createDevice();
    auto kernel = Doubler{};
    EXPECT_THROW(device->bindBuffer(kernel.data), DeviceError);
}

} // namespace
