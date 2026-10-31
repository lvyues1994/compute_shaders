// The transfer queue: copies on a second timeline and the waits between
// the two.

#include <vector>

#include "equivalence.hpp"
#include "kernels/memory_kernels.hpp"
#include "kernels/texture_kernels.hpp"

namespace {

using namespace tc;
using namespace tc::testing_support;

constexpr int kWords = 256;

template <class In, class Out>
void slowCopy(Device &device, In &in, Out &out, BufferResource<uint> &sink, uint const rounds) {
    auto kernel = tests::SlowCopy{};
    kernel.in.attach(in);
    kernel.out.attach(out);
    kernel.sink.attach(sink);
    kernel.addend = 0u;
    kernel.rounds = rounds;
    device.useKernel(kernel);
    device.bindBuffer(kernel.in);
    device.bindBuffer(kernel.out);
    device.bindBuffer(kernel.sink);
    device.setUniform(kernel.addend);
    device.setUniform(kernel.rounds);
    device.execute(kernel, uvec3{static_cast<uint>(kWords) / 64u, 1u, 1u});
}

struct TransferResults {
    std::vector<uint> copied;
    std::vector<vec4> texels;
};

// Words and texels reach the device through the transfer queue; the
// device waits for them before a kernel reads them.
TransferResults copyThroughTransfers(Device &device) {
    auto source = device.createBuffer<uint>(kWords);
    for (int i = 0; i < kWords; ++i) {
        source[i] = static_cast<uint>(i * 11 + 5);
    }
    device.uploadBuffer(source);
    auto words = device.createDeviceBuffer<uint>(kWords);
    auto pool = device.createImage<gpu::RGBA8, Dim::D2Array>(ivec3{16, 8, 2});
    auto &transfers = device.transfers();
    transfers.waitFor(device.submit()); // the upload of `source`
    transfers.copyBuffer(source, 0, words, 0, kWords * sizeof(uint));
    transfers.copyBufferToImage(source, 0, pool, 0, 1);
    auto const copied = transfers.submit();
    EXPECT_GT(copied.sequence(), 0u);
    device.waitFor(copied);

    auto out = device.createBuffer<uint>(kWords);
    auto sink = BufferResource<uint>{kWords};
    device.uploadBuffer(out);
    device.uploadBuffer(sink);
    slowCopy(device, words, out, sink, 4u);
    auto texels = device.createBuffer<vec4>(16 * 8 * 2 * 2);
    device.uploadBuffer(texels);
    auto read = tests::ReadPool{};
    read.pool.attach(pool);
    read.out.attach(texels);
    read.level = 0;
    device.useKernel(read);
    device.bindTexture(read.pool);
    device.bindBuffer(read.out);
    device.setUniform(read.level);
    device.execute(read, uvec3{2u, 2u, 2u});
    device.finish();
    return TransferResults{contents(out), contents(texels)};
}

TEST_P(Equivalence, TransferQueueCopies) {
    auto const [expected, actual] = runBoth<TransferResults>(copyThroughTransfers);
    EXPECT_EQ(expected.copied, actual.copied);
    for (int i = 0; i < kWords; ++i) {
        ASSERT_EQ(actual.copied[i], static_cast<uint>(i * 11 + 5)) << "word " << i;
    }
    // ReadPool writes two values per texel; layer 1 holds the copy.
    auto const fetched = actual.texels[(128 + 37) * 2];
    auto const want = unpackUnorm4x8(static_cast<uint>(37 * 11 + 5));
    for (int c = 0; c < 4; ++c) {
        EXPECT_NEAR(fetched[c], want[c], 1e-6f);
    }
}

struct OrderResults {
    std::vector<uint> before;
    std::vector<uint> after;
};

// The device reads a buffer slowly; the transfer queue waits for that
// submission before overwriting the buffer, and the device waits for the
// copy before reading again.
OrderResults ordersBothWays(Device &device) {
    auto oldWords = BufferResource<uint>{kWords};
    auto newWords = BufferResource<uint>{kWords};
    for (int i = 0; i < kWords; ++i) {
        oldWords[i] = static_cast<uint>(i);
        newWords[i] = static_cast<uint>(1000 + i);
    }
    device.uploadBuffer(oldWords);
    device.uploadBuffer(newWords);
    auto words = device.createDeviceBuffer<uint>(kWords);
    device.copyBuffer(oldWords, 0, words, 0, kWords * sizeof(uint));
    auto before = device.createBuffer<uint>(kWords);
    auto after = device.createBuffer<uint>(kWords);
    auto sink = BufferResource<uint>{kWords};
    device.uploadBuffer(before);
    device.uploadBuffer(after);
    device.uploadBuffer(sink);
    slowCopy(device, words, before, sink, 1u << 14);
    auto const reading = device.submit();
    auto &transfers = device.transfers();
    transfers.waitFor(reading);
    transfers.copyBuffer(newWords, 0, words, 0, kWords * sizeof(uint));
    device.waitFor(transfers.submit());
    slowCopy(device, words, after, sink, 4u);
    device.finish();
    return OrderResults{contents(before), contents(after)};
}

TEST_P(Equivalence, TransferQueueOrdersBothWays) {
    auto const [expected, actual] = runBoth<OrderResults>(ordersBothWays);
    for (auto const *result : {&expected, &actual}) {
        for (int i = 0; i < kWords; ++i) {
            ASSERT_EQ(result->before[i], static_cast<uint>(i)) << "the copy overtook the read of word " << i;
            ASSERT_EQ(result->after[i], static_cast<uint>(1000 + i)) << "the read missed the copy of word " << i;
        }
    }
}

TEST_P(Equivalence, TransferQueueReportsItsQueue) {
    EXPECT_FALSE(cpu->info().separateTransfers);
    auto const info = gpu->info();
    if (info.backend != "vulkan") {
        EXPECT_FALSE(info.separateTransfers) << info.backend;
    }
    auto &transfers = gpu->transfers();
    EXPECT_EQ(&transfers, &gpu->transfers()) << "one queue per device";
    auto const nothing = transfers.submit();
    nothing.wait();
    EXPECT_TRUE(nothing.isComplete());
    // A submission of another device is waited for on the host.
    auto other = cpu::createDevice();
    gpu->waitFor(other->submit());
    transfers.waitFor(other->submit());
}

// With a queue of its own, a copy completes while a long dispatch runs.
TEST_P(Equivalence, TransferQueueRunsNextToDispatches) {
    if (not gpu->info().separateTransfers) {
        GTEST_SKIP() << gpu->info().deviceName << " has no second queue";
    }
    auto in = gpu->createDeviceBuffer<uint>(kWords);
    auto out = gpu->createDeviceBuffer<uint>(kWords);
    auto sink = BufferResource<uint>{kWords};
    gpu->uploadBuffer(sink);
    auto zero = BufferResource<uint>{kWords};
    gpu->uploadBuffer(zero);
    gpu->copyBuffer(zero, 0, in, 0, kWords * sizeof(uint));
    slowCopy(*gpu, in, out, sink, 1u << 24);
    auto const busy = gpu->submit();
    auto large = gpu->createBuffer<uint>(4 << 20);
    auto copy = gpu->createDeviceBuffer<uint>(4 << 20);
    gpu->uploadBuffer(large);
    auto &transfers = gpu->transfers();
    transfers.copyBuffer(large, 0, copy, 0, std::size_t{16} << 20);
    auto const copied = transfers.submit();
    copied.wait();
    EXPECT_FALSE(busy.isComplete()) << "the copy waited for the dispatch";
    busy.wait();
}

} // namespace
