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

TEST(Mat, IsColumnMajorLikeGlsl) {
    auto const m = mat2{1.0f, 2.0f, 3.0f, 4.0f}; // columns (1, 2) and (3, 4)
    EXPECT_EQ(m[1], (vec2{3.0f, 4.0f}));
    EXPECT_EQ((m * vec2{1.0f, 0.0f}), (vec2{1.0f, 2.0f}));
    EXPECT_EQ((vec2{1.0f, 0.0f} * m), (vec2{1.0f, 3.0f}));
    EXPECT_EQ((mat3{2.0f} * vec3{1.0f, 2.0f, 3.0f}), (vec3{2.0f, 4.0f, 6.0f}));
    EXPECT_EQ(transpose(m), (mat2{1.0f, 3.0f, 2.0f, 4.0f}));
    EXPECT_FLOAT_EQ(determinant(m), -2.0f);
    auto const product = mat2x3{vec3{1.0f, 0.0f, 0.0f}, vec3{0.0f, 1.0f, 0.0f}} * mat2{1.0f, 2.0f, 3.0f, 4.0f};
    EXPECT_EQ(product[1], (vec3{3.0f, 4.0f, 0.0f}));
    EXPECT_EQ(mat3{mat2{5.0f}}, (mat3{5.0f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 1.0f}));
}

TEST(Mat, InvertsAndMultiplies) {
    auto const m3 = mat3{2.0f, 1.0f, 0.0f, 0.0f, 3.0f, 1.0f, 1.0f, 0.0f, 4.0f};
    auto const i3 = m3 * inverse(m3);
    auto const m4 = mat4{4.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 1.0f, 0.0f,
                         1.0f, 0.0f, 5.0f, 1.0f, 0.0f, 2.0f, 0.0f, 6.0f};
    auto const i4 = m4 * inverse(m4);
    for (int c = 0; c < 4; ++c) {
        for (int r = 0; r < 4; ++r) {
            if (c < 3 and r < 3) {
                EXPECT_NEAR(i3[c][r], c == r ? 1.0f : 0.0f, 1e-5f);
            }
            EXPECT_NEAR(i4[c][r], c == r ? 1.0f : 0.0f, 1e-5f);
        }
    }
    EXPECT_NEAR(determinant(m4) * determinant(inverse(m4)), 1.0f, 1e-5f);
    EXPECT_EQ(outerProduct(vec3{1.0f, 2.0f, 3.0f}, vec2{1.0f, 10.0f})[1], (vec3{10.0f, 20.0f, 30.0f}));
}

TEST(Packing, MatchesGlslDefinitions) {
    EXPECT_EQ(packUnorm4x8(vec4{0.0f, 1.0f, 0.5f, 2.0f}), 0xff80ff00u);
    EXPECT_EQ(unpackUnorm4x8(0xff80ff00u), (vec4{0.0f, 1.0f, 128.0f / 255.0f, 1.0f}));
    EXPECT_EQ(packSnorm4x8(vec4{-1.0f, 1.0f, 0.0f, -0.5f}), 0xc0007f81u);
    EXPECT_EQ(unpackHalf2x16(packHalf2x16(vec2{1.5f, -0.000061035156f})), (vec2{1.5f, -0.000061035156f}));
    EXPECT_EQ(packHalf2x16(vec2{65504.0f, 0.0f}), 0x00007bffu);
    EXPECT_EQ(bitfieldExtract(0xabcdu, 4, 8), 0xbcu);
    EXPECT_EQ(bitfieldExtract(std::int32_t{0x0f0}, 4, 4), -1);
    EXPECT_EQ(bitfieldInsert(0xffffu, 0u, 4, 8), 0xf00fu);
    EXPECT_EQ(bitCount(uvec2{0xffu, 0x10u}), (ivec2{8, 1}));
    EXPECT_EQ(findLSB(0x18u), 3);
    EXPECT_EQ(findMSB(std::int32_t{-1}), -1);
    EXPECT_EQ(findMSB(0x18u), 4);
    EXPECT_EQ(bitfieldReverse(1u), 0x80000000u);
}

TEST(Texture, SamplesLikeOpenGl) {
    auto pixels = BufferResource<cpu::R8, Dim::D2>{ivec2{2, 2}};
    pixels[ivec2{0, 0}] = cpu::R8{{0}};
    pixels[ivec2{1, 0}] = cpu::R8{{255}};
    pixels[ivec2{0, 1}] = cpu::R8{{51}};
    pixels[ivec2{1, 1}] = cpu::R8{{102}};
    auto nearest = TextureBinding<gpu::R8, Dim::D2, cpu::R8, 0>{};
    auto linear = TextureBinding<gpu::R8, Dim::D2, cpu::R8, 1, Filter::Linear, Wrap::Repeat>{};
    nearest.attach(pixels);
    linear.attach(pixels);
    EXPECT_FLOAT_EQ(texelFetch(nearest, ivec2{1, 0}, 0).x, 1.0f);
    EXPECT_FLOAT_EQ(texture(nearest, vec2{0.9f, 0.9f}).x, 0.4f);
    EXPECT_FLOAT_EQ(texture(nearest, vec2{1.9f, 0.1f}).x, 1.0f); // clamped
    EXPECT_NEAR(texture(linear, vec2{0.5f, 0.25f}).x, 0.5f, 1e-6f);
    EXPECT_NEAR(texture(linear, vec2{0.5f, 0.5f}).x, (0.0f + 1.0f + 0.2f + 0.4f) / 4.0f, 1e-6f);
    EXPECT_NEAR(texture(linear, vec2{0.0f, 0.25f}).x, 0.5f, 1e-6f); // wraps between columns 1 and 0
    EXPECT_EQ(textureSize(linear, 0), (ivec2{2, 2}));
    EXPECT_EQ(texture(nearest, vec2{0.25f, 0.25f}).w, 1.0f);
}

TEST(BufferResource, WrapsExternalMemory) {
    auto backing = std::make_shared<std::vector<float>>(8, 1.5f);
    auto watcher = std::weak_ptr<std::vector<float>>{backing};
    auto memory = ExternalMemory{reinterpret_cast<std::byte *>(backing->data()), 8 * sizeof(float), backing};
    backing.reset();
    {
        auto resource = BufferResource<float>::wrap(8, std::move(memory), ResourceIdentity::create());
        EXPECT_TRUE(resource.usesExternalMemory());
        EXPECT_FLOAT_EQ(resource[3], 1.5f);
        resource[3] = 2.0f;
        auto const copy = resource;
        EXPECT_FALSE(copy.usesExternalMemory());
        EXPECT_FLOAT_EQ(copy[3], 2.0f);
        auto const moved = std::move(resource);
        EXPECT_TRUE(moved.usesExternalMemory());
        EXPECT_FALSE(watcher.expired());
    }
    EXPECT_TRUE(watcher.expired());
}

struct Inner {
    vec2 uv;
    std::uint32_t id;
};

enum class Kind : std::uint32_t { Rock = 1, Paper = 7 };

struct Body {
    vec3 position;
    float mass;
    mat3 frame;
    std::array<float, 3> weights;
    Inner inner;
    bool alive;
    Kind kind;
    double precise;
    std::int32_t offset;
};

Body sampleBody(float const seed) {
    auto body = Body{};
    body.position = vec3{seed, seed + 1.0f, seed + 2.0f};
    body.mass = 3.5f * seed;
    body.frame = mat3{vec3{1.0f, 2.0f, 3.0f}, vec3{4.0f, 5.0f, 6.0f}, vec3{7.0f, 8.0f, seed}};
    body.weights = {0.25f, 0.5f, seed};
    body.inner = Inner{vec2{-seed, 9.0f}, 42u};
    body.alive = true;
    body.kind = Kind::Paper;
    body.precise = 1.0 / 3.0 + seed;
    body.offset = -17;
    return body;
}

void expectSame(Body const &a, Body const &b) {
    EXPECT_EQ(a.position, b.position);
    EXPECT_EQ(a.mass, b.mass);
    EXPECT_EQ(a.frame, b.frame);
    EXPECT_EQ(a.weights, b.weights);
    EXPECT_EQ(a.inner.uv, b.inner.uv);
    EXPECT_EQ(a.inner.id, b.inner.id);
    EXPECT_EQ(a.alive, b.alive);
    EXPECT_EQ(a.kind, b.kind);
    EXPECT_EQ(a.precise, b.precise);
    EXPECT_EQ(a.offset, b.offset);
}

TEST(Table, PacksFieldsWithoutPadding) {
    using Fields = detail::TableFields<Body>;
    static_assert(Fields::count == 9);
    static_assert(Fields::offset<0> == 0 and Fields::offset<1> == 3 and Fields::offset<2> == 4);
    static_assert(Fields::offset<3> == 13 and Fields::offset<4> == 16 and Fields::offset<5> == 19);
    static_assert(Fields::offset<6> == 20 and Fields::offset<7> == 21 and Fields::offset<8> == 23);
    static_assert(detail::Packing<Body>::words == 24);
    static_assert(detail::Packing<vec3>::words == 3 and detail::Packing<mat3>::words == 9);
    EXPECT_LT(detail::Packing<Body>::words * sizeof(std::uint32_t), sizeof(Body)); // 96 bytes vs 128

    auto table = TableResource<vec3>{5};
    EXPECT_EQ(table.words().size(), 4u + 15u);
    EXPECT_EQ(table.words()[0], 5u);
}

TEST(Table, RoundTripsEveryFieldType) {
    auto table = TableResource<Body>{4};
    EXPECT_EQ(table.size(), 4u);
    EXPECT_EQ(table.words()[0], 4u);
    for (std::size_t i = 0; i < table.size(); ++i) {
        table.set(i, sampleBody(static_cast<float>(i)));
    }
    for (std::size_t i = 0; i < table.size(); ++i) {
        expectSame(table.get(i), sampleBody(static_cast<float>(i)));
    }
    // Record 2 starts right after the header and two 24-word records.
    auto const position = vec3{2.0f, 3.0f, 4.0f};
    EXPECT_EQ(std::memcmp(&table.words()[4 + 2 * 24], &position, sizeof(float) * 3), 0);
}

// Field writes, partial writes and whole-element copies through a binding.
struct TC_KERNEL TableEdits {
    static constexpr char fileLocation[] = "table_edits";
    uvec3 local_size{2, 1, 1};

    TableBinding<Body, 0> bodies;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i == 0u) {
            bodies[1].mass = 5.0f;
            bodies[1].position.y -= 1.0f;
            bodies[1].inner.uv.x += 0.5f;
        } else {
            bodies[3] = bodies[2];
            bodies[3].offset = int(bodies.length());
        }
    }
};

TEST(Table, KernelsEditElementsInPlace) {
    auto device = cpu::createDevice(cpu::CpuDeviceConfig{2});
    auto table = TableResource<Body>{4};
    for (std::size_t i = 0; i < table.size(); ++i) {
        table.set(i, sampleBody(static_cast<float>(i)));
    }
    auto kernel = TableEdits{};
    kernel.bodies.attach(table);
    device->useKernel(kernel);
    device->bindTable(kernel.bodies);
    device->execute(kernel, uvec3{1u, 1u, 1u});
    auto const edited = table.get(1);
    EXPECT_EQ(edited.mass, 5.0f);
    EXPECT_EQ(edited.position, (vec3{1.0f, 1.0f, 3.0f}));
    EXPECT_EQ(edited.inner.uv.x, -0.5f);
    auto expected = sampleBody(2.0f);
    expected.offset = 4;
    expectSame(table.get(3), expected);
    expectSame(table.get(0), sampleBody(0.0f));
}

TEST(Table, WritesBackOnlyChangedWords) {
    auto table = TableResource<Body>{2};
    table.set(1, sampleBody(1.0f));
    auto &cache = detail::TableCache<Body, Aos>::forThisThread();
    cache.element(table, 1).mass = 9.0f;
    table.words()[4 + 24 + 0] = 0x40400000u; // meanwhile another workgroup sets position.x = 3.0f
    detail::flushTables(false);
    EXPECT_EQ(table.get(1).mass, 9.0f);
    EXPECT_EQ(table.get(1).position.x, 3.0f);
}

TEST(Codec, RoundTripsOnTheHost) {
    static_assert(detail::Packing<Encoded<vec4, Unorm8x4>>::words == 1);
    static_assert(detail::Packing<Encoded<mat4, Affine3x4>>::words == 12);
    static_assert(detail::Packing<Encoded<vec4, Half4>>::words == 2);
    static_assert(detail::IsLossy<Encoded<vec2, Half2>>::value and not detail::IsLossy<Body>::value);

    auto const color = Unorm8x4::decode(Unorm8x4::encode(vec4{0.2f, 0.4f, 0.6f, 1.0f}));
    EXPECT_NEAR(color.y, 0.4f, 0.5f / 255.0f);
    for (int octant = 0; octant < 8; ++octant) {
        auto const n = normalize(vec3{octant & 1 ? -0.3f : 0.3f, octant & 2 ? -0.8f : 0.8f, octant & 4 ? -0.5f : 0.5f});
        auto const back = OctahedralNormal::decode(OctahedralNormal::encode(n));
        for (int c = 0; c < 3; ++c) {
            EXPECT_NEAR(back[c], n[c], 1e-4f) << "octant " << octant;
        }
    }
    auto const transform = mat4{vec4{1.0f, 2.0f, 3.0f, 0.0f}, vec4{4.0f, 5.0f, 6.0f, 0.0f},
                                vec4{7.0f, 8.0f, 9.0f, 0.0f}, vec4{10.0f, 11.0f, 12.0f, 1.0f}};
    EXPECT_EQ(Affine3x4::decode(Affine3x4::encode(transform)), transform);
}

struct Tinted {
    Encoded<vec4, Unorm8x4> color;
    float weight;
};

// A codec field written and read again in one CPU invocation reads back
// quantized, as the GPU's store and load would.
TEST(Table, CodecFieldsReadBackQuantized) {
    auto table = TableResource<Tinted>{1};
    auto &cache = detail::TableCache<Tinted, Aos>::forThisThread();
    cache.element(table, 0).color = vec4{0.3f};
    auto const reread = cache.element(table, 0).color.value.x;
    EXPECT_NE(reread, 0.3f);
    EXPECT_EQ(reread, Unorm8x4::decode(Unorm8x4::encode(vec4{0.3f})).x);
    detail::flushTables(false);
    EXPECT_EQ(table.words()[4], packUnorm4x8(vec4{0.3f}));
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

struct TC_KERNEL DivergentBarrier {
    static constexpr char fileLocation[] = "divergent_barrier";
    static constexpr bool usesBarrier = true;
    uvec3 local_size{4, 1, 1};

    BufferBinding<uint, 0> out;

    void main() {
        if (gl_LocalInvocationIndex == 2u) {
            return;
        }
        barrier();
        out[gl_LocalInvocationIndex] = 1u;
    }
};

struct TC_KERNEL UndeclaredBarrier {
    static constexpr char fileLocation[] = "undeclared_barrier";
    uvec3 local_size{4, 1, 1};

    BufferBinding<uint, 0> out;

    void main() {
        barrier();
        out[gl_LocalInvocationIndex] = 1u;
    }
};

template <class K>
void runOnCpu(K &kernel) {
    auto device = cpu::createDevice(cpu::CpuDeviceConfig{1});
    auto out = BufferResource<uint>{4};
    kernel.out.attach(out);
    device->useKernel(kernel);
    device->bindBuffer(kernel.out);
    device->execute(kernel, uvec3{1u, 1u, 1u});
}

TEST(CpuDeviceDeathTest, DetectsInvocationsSkippingABarrier) {
    auto kernel = DivergentBarrier{};
    EXPECT_DEATH(runOnCpu(kernel), "not reached by every invocation");
}

TEST(CpuDeviceDeathTest, RequiresUsesBarrier) {
    auto kernel = UndeclaredBarrier{};
    EXPECT_DEATH(runOnCpu(kernel), "usesBarrier");
}

TEST(CpuDevice, RejectsUnattachedBindings) {
    auto device = cpu::createDevice();
    auto kernel = Doubler{};
    EXPECT_THROW(device->bindBuffer(kernel.data), DeviceError);
}

} // namespace
