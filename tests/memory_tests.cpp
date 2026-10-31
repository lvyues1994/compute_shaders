// Device-only buffers, copies between buffers, heaps and aliasing.

#include <cstring>
#include <vector>

#include "equivalence.hpp"
#include "kernels/dispatch_kernels.hpp"
#include "kernels/memory_kernels.hpp"
#include "kernels/texture_kernels.hpp"

namespace {

using namespace tc;
using namespace tc::testing_support;

constexpr int kWords = 256;

void fill(Device &device, BufferBinding<uint, 0> const &target, uint const seed) {
    auto kernel = tests::FillWords{};
    kernel.out = target;
    kernel.seed = seed;
    device.useKernel(kernel);
    device.bindBuffer(kernel.out);
    device.setUniform(kernel.seed);
    device.execute(kernel, uvec3{static_cast<uint>(kWords + 63) / 64u, 1u, 1u});
}

template <class In, class Out>
void slowCopy(Device &device, In &in, Out &out, BufferResource<uint> &sink, uint const addend) {
    auto kernel = tests::SlowCopy{};
    kernel.in.attach(in);
    kernel.out.attach(out);
    kernel.sink.attach(sink);
    kernel.addend = addend;
    kernel.rounds = 512u;
    device.useKernel(kernel);
    device.bindBuffer(kernel.in);
    device.bindBuffer(kernel.out);
    device.bindBuffer(kernel.sink);
    device.setUniform(kernel.addend);
    device.setUniform(kernel.rounds);
    device.execute(kernel, uvec3{static_cast<uint>(kWords + 63) / 64u, 1u, 1u});
}

TEST_P(Equivalence, DeviceBuffersHaveNoHostCopy) {
    for (auto *device : {cpu.get(), gpu.get()}) {
        auto const name = device->info().backend;
        auto buffer = device->createDeviceBuffer<uint>(kWords);
        ASSERT_TRUE(buffer) << name;
        EXPECT_EQ(buffer.size(), static_cast<std::size_t>(kWords)) << name;
        EXPECT_EQ(buffer.hostView()->hasHostData(), device == cpu.get()) << name << ": host data only on the CPU";
        auto const alias = buffer;
        EXPECT_EQ(alias.identity().key(), buffer.identity().key()) << name << ": copies are the same buffer";
        EXPECT_FALSE(buffer.placement().heap) << name;
        if (device != cpu.get()) {
            EXPECT_THROW(device->uploadBuffer(*buffer.hostView()), DeviceError) << name;
            EXPECT_THROW(device->downloadBuffer(*buffer.hostView()), DeviceError) << name;
        }
        EXPECT_THROW(device->createDeviceBuffer<uint>(0), DeviceError) << name;
    }
}

// Words go host buffer -> device buffer -> kernel -> device buffer -> host
// buffer, an indirect dispatch reads its counts from a device buffer, and
// a device buffer fills a device image.
struct CopyResults {
    std::vector<uint> words;
    std::vector<float> indirect;
    std::vector<vec4> texels;
};

CopyResults copyBetweenKinds(Device &device) {
    auto source = BufferResource<uint>{kWords};
    for (int i = 0; i < kWords; ++i) {
        source[i] = static_cast<uint>(i * 7 + 3);
    }
    device.uploadBuffer(source);
    auto staged = device.createDeviceBuffer<uint>(kWords + 64);
    device.copyBuffer(source, 0, staged, 64 * sizeof(uint), kWords * sizeof(uint));
    auto computed = device.createDeviceBuffer<uint>(kWords);
    auto sink = BufferResource<uint>{kWords};
    device.uploadBuffer(sink);
    auto kernel = tests::SlowCopy{};
    kernel.in.attach(staged, 64, kWords);
    kernel.out.attach(computed);
    kernel.sink.attach(sink);
    kernel.addend = 1000u;
    kernel.rounds = 4u;
    device.useKernel(kernel);
    device.bindBuffer(kernel.in);
    device.bindBuffer(kernel.out);
    device.bindBuffer(kernel.sink);
    device.setUniform(kernel.addend);
    device.setUniform(kernel.rounds);
    device.execute(kernel, uvec3{kWords / 64u, 1u, 1u});
    auto result = BufferResource<uint>{kWords};
    device.uploadBuffer(result);
    device.copyBuffer(computed, 0, result, 0, kWords * sizeof(uint));
    device.downloadBuffer(result);

    auto counts = BufferResource<DispatchIndirectCommand>{1};
    counts[0] = DispatchIndirectCommand{3, 1, 1};
    device.uploadBuffer(counts);
    auto arguments = device.createDeviceBuffer<DispatchIndirectCommand>(2);
    device.copyBuffer(counts, 0, arguments, sizeof(DispatchIndirectCommand), sizeof(DispatchIndirectCommand));
    auto spheres = BufferResource<vec4>{256};
    auto visible = BufferResource<uint>{256};
    auto count = BufferResource<uint>{1};
    auto shaded = BufferResource<float>{256};
    for (int i = 0; i < 256; ++i) {
        spheres[i] = vec4{0.0f, 0.0f, 0.0f, static_cast<float>(i)};
        visible[i] = static_cast<uint>(i);
    }
    count[0] = 150;
    for (auto *buffer : {&visible, &count}) {
        device.uploadBuffer(*buffer);
    }
    device.uploadBuffer(spheres);
    device.uploadBuffer(shaded);
    auto shade = tests::ShadeVisible{};
    shade.spheres.attach(spheres);
    shade.visible.attach(visible);
    shade.count.attach(count);
    shade.out.attach(shaded);
    device.useKernel(shade);
    device.bindBuffer(shade.spheres);
    device.bindBuffer(shade.visible);
    device.bindBuffer(shade.count);
    device.bindBuffer(shade.out);
    device.executeIndirect(shade, arguments, 1);
    device.downloadBuffer(shaded);

    auto texels = device.createDeviceBuffer<uint>(16 * 8);
    device.copyBuffer(source, 0, texels, 0, 16 * 8 * sizeof(uint));
    auto pool = device.createImage<gpu::RGBA8, Dim::D2Array>(ivec3{16, 8, 1});
    device.copyBufferToImage(texels, 0, pool, 0);
    auto read = tests::ReadPool{};
    auto out = BufferResource<vec4>{16 * 8 * 2};
    device.uploadBuffer(out);
    read.pool.attach(pool);
    read.out.attach(out);
    read.level = 0;
    device.useKernel(read);
    device.bindTexture(read.pool);
    device.bindBuffer(read.out);
    device.setUniform(read.level);
    device.execute(read, uvec3{2u, 2u, 1u});
    device.downloadBuffer(out);
    return CopyResults{contents(result), contents(shaded), contents(out)};
}

TEST_P(Equivalence, CopyBetweenBufferKinds) {
    auto const [expected, actual] = runBoth<CopyResults>(copyBetweenKinds);
    EXPECT_EQ(expected.words, actual.words);
    for (int i = 0; i < kWords; ++i) {
        ASSERT_EQ(actual.words[i], static_cast<uint>(i * 7 + 3 + 1000)) << "word " << i;
    }
    EXPECT_EQ(expected.indirect, actual.indirect);
    EXPECT_EQ(actual.indirect[149], 149.0f * 2.0f + 3.0f) << "three workgroups, from the device buffer";
    EXPECT_EQ(actual.indirect[200], 0.0f);
    ASSERT_EQ(expected.texels.size(), actual.texels.size());
    for (std::size_t i = 0; i < actual.texels.size(); i += 2) {
        auto const packed = static_cast<uint>((i / 2) * 7 + 3);
        auto const want = unpackUnorm4x8(packed);
        for (int c = 0; c < 4; ++c) {
            EXPECT_NEAR(actual.texels[i][c], want[c], 1e-6f) << "texel " << i / 2;
        }
    }
}

TEST_P(Equivalence, CopyBufferRules) {
    for (auto *device : {cpu.get(), gpu.get()}) {
        auto const name = device->info().backend;
        auto a = device->createDeviceBuffer<uint>(16);
        auto b = BufferResource<uint>{16};
        device->uploadBuffer(b);
        EXPECT_THROW(device->copyBuffer(a, 2, b, 0, 4), DeviceError) << name << ": misaligned";
        EXPECT_THROW(device->copyBuffer(a, 0, b, 0, 0), DeviceError) << name << ": empty";
        EXPECT_THROW(device->copyBuffer(a, 0, b, 4, 64), DeviceError) << name << ": past the end";
        EXPECT_THROW(device->copyBuffer(a, 0, a, 8, 16), DeviceError) << name << ": overlapping";
        EXPECT_NO_THROW(device->copyBuffer(a, 0, a, 32, 32)) << name;
    }
}

// Two buffers at the same place in a heap: the second one's writes wait
// for the first one's slow reads, and they really share the memory.
struct AliasResults {
    std::vector<uint> first;
    std::vector<uint> second;
    std::vector<uint> firstAgain; // read after the second was written: undefined contents
};

AliasResults aliasedBuffers(Device &device) {
    auto const size = device.bufferPlacement<uint>(kWords);
    auto heap = device.createHeap(size.bytes + size.alignment);
    auto a = device.createDeviceBuffer<uint>(kWords, Placement{heap, 0});
    auto b = device.createDeviceBuffer<uint>(kWords, Placement{heap, 0});
    auto sink = BufferResource<uint>{kWords};
    device.uploadBuffer(sink);
    auto outA = device.createBuffer<uint>(kWords);
    auto outB = device.createBuffer<uint>(kWords);
    auto outAgain = device.createBuffer<uint>(kWords);
    for (auto *buffer : {&outA, &outB, &outAgain}) {
        device.uploadBuffer(*buffer);
    }
    auto target = BufferBinding<uint, 0>{};
    target.attach(a);
    fill(device, target, 3u);
    slowCopy(device, a, outA, sink, 0u);
    target.attach(b);
    fill(device, target, 5u); // overwrites what the slow copy is reading, after it
    slowCopy(device, b, outB, sink, 0u);
    slowCopy(device, a, outAgain, sink, 0u);
    device.finish();
    return AliasResults{contents(outA), contents(outB), contents(outAgain)};
}

TEST_P(Equivalence, AliasedBuffersShareMemoryInOrder) {
    auto const [expected, actual] = runBoth<AliasResults>(aliasedBuffers);
    for (auto const *result : {&expected, &actual}) {
        for (int i = 0; i < kWords; ++i) {
            ASSERT_EQ(result->first[i], static_cast<uint>(i) * 3u + 1u) << "word " << i;
            ASSERT_EQ(result->second[i], static_cast<uint>(i) * 5u + 1u) << "word " << i;
        }
    }
    for (int i = 0; i < kWords; ++i) {
        ASSERT_EQ(expected.firstAgain[i], 0xCDCDCDCDu) << "the CPU fills a reused alias with garbage";
        ASSERT_EQ(actual.firstAgain[i], static_cast<uint>(i) * 5u + 1u) << "the GPU buffers share memory";
    }
}

// A buffer and an image at the same place in a heap, used in turn.
struct ImageAliasResults {
    std::vector<vec4> colour;
    std::vector<uint> words;
    std::vector<vec4> colourAgain;
};

void paint(Device &device, DeviceImage<gpu::RGBA16F, Dim::D2, cpu::RGBA32F> const &image, BufferResource<vec4> &out,
           float const seed) {
    auto fillColour = tests::FillColour{};
    fillColour.image.attach(image);
    fillColour.seed = seed;
    device.useKernel(fillColour);
    device.bindImage(fillColour.image);
    device.setUniform(fillColour.seed);
    device.execute(fillColour, uvec3{2u, 2u, 1u});
    auto read = tests::ReadColour{};
    read.image.attach(image);
    read.out.attach(out);
    device.useKernel(read);
    device.bindTexture(read.image);
    device.bindBuffer(read.out);
    device.execute(read, uvec3{2u, 2u, 1u});
}

ImageAliasResults aliasedImage(Device &device) {
    auto const imageSize = device.imagePlacement<gpu::RGBA16F, Dim::D2, cpu::RGBA32F>(ivec2{16, 16});
    auto const bufferSize = device.bufferPlacement<uint>(kWords);
    auto heap = device.createHeap(std::max(imageSize.bytes, bufferSize.bytes));
    auto image = device.createImage<gpu::RGBA16F, Dim::D2, cpu::RGBA32F>(ivec2{16, 16}, 1, Placement{heap, 0});
    auto words = device.createDeviceBuffer<uint>(kWords, Placement{heap, 0});
    auto colour = device.createBuffer<vec4>(256);
    auto colourAgain = device.createBuffer<vec4>(256);
    auto out = device.createBuffer<uint>(kWords);
    auto sink = BufferResource<uint>{kWords};
    for (auto *buffer : {&colour, &colourAgain}) {
        device.uploadBuffer(*buffer);
    }
    device.uploadBuffer(out);
    device.uploadBuffer(sink);
    paint(device, image, colour, 1.0f);
    auto target = BufferBinding<uint, 0>{};
    target.attach(words);
    fill(device, target, 7u);
    slowCopy(device, words, out, sink, 0u);
    paint(device, image, colourAgain, 2.0f);
    device.finish();
    return ImageAliasResults{contents(colour), contents(out), contents(colourAgain)};
}

TEST_P(Equivalence, ImagesAndBuffersAlias) {
    EXPECT_TRUE(cpu->info().placedImages);
    EXPECT_EQ(gpu->info().placedImages, gpu->info().backend == "vulkan");
    auto const [expected, actual] = runBoth<ImageAliasResults>(aliasedImage);
    for (auto const *result : {&expected, &actual}) {
        for (int y = 0; y < 16; ++y) {
            for (int x = 0; x < 16; ++x) {
                ASSERT_EQ(result->colour[y * 16 + x], (vec4{float(x), float(y), 1.0f, 1.0f}));
                ASSERT_EQ(result->colourAgain[y * 16 + x], (vec4{float(x), float(y), 2.0f, 1.0f}));
            }
        }
        for (int i = 0; i < kWords; ++i) {
            ASSERT_EQ(result->words[i], static_cast<uint>(i) * 7u + 1u);
        }
    }
}

TEST_P(Equivalence, PlacementRules) {
    for (auto *device : {cpu.get(), gpu.get()}) {
        auto const name = device->info().backend;
        auto const size = device->bufferPlacement<vec4>(100);
        EXPECT_GE(size.bytes, 100 * sizeof(vec4)) << name;
        EXPECT_GE(size.alignment, 16u) << name;
        EXPECT_EQ(size.alignment & (size.alignment - 1), 0u) << name << ": a power of two";
        auto const image = device->imagePlacement<gpu::RGBA8, Dim::D2>(ivec2{32, 32}, 0);
        EXPECT_GE(image.bytes, 32u * 32u * 4u) << name;
        EXPECT_THROW(device->createHeap(0), DeviceError) << name;
        auto heap = device->createHeap(2 * size.bytes + size.alignment);
        EXPECT_EQ(heap.size(), 2 * size.bytes + size.alignment) << name;
        EXPECT_THROW(device->createDeviceBuffer<vec4>(100, Placement{heap, size.alignment / 2}), DeviceError)
            << name << ": misaligned";
        EXPECT_THROW(device->createDeviceBuffer<vec4>(100, Placement{heap, 2 * size.alignment + size.bytes}), DeviceError)
            << name << ": past the end";
        EXPECT_THROW(device->createDeviceBuffer<vec4>(100, Placement{MemoryHeap{}, 0}), DeviceError) << name;
        auto placed = device->createDeviceBuffer<vec4>(100, Placement{heap, size.alignment});
        EXPECT_EQ(placed.placement().offset, size.alignment) << name;
        EXPECT_EQ(placed.placement().heap.identity().key(), heap.identity().key()) << name;
        heap = MemoryHeap{};
        EXPECT_TRUE(placed.placement().heap) << name << ": the placed buffer keeps its heap";
    }
}

} // namespace
