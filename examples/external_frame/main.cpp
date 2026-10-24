// Android only: the zero-copy path for platform frames. A YUV and an RGBA
// AHardwareBuffer are filled the way a camera or decoder would fill them,
// sampled in place by the kernel, and converted into a createBuffer() result
// the CPU reads without a copy.

#include <android/hardware_buffer.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include <examples/cli.hpp>
#include <examples/ppm.hpp>
#include <nv12_to_rgba/convert.hpp>
#include <tc/devices.hpp>

#include "external_frame.hpp"

namespace {

using HardwareBuffer = std::unique_ptr<AHardwareBuffer, void (*)(AHardwareBuffer *)>;

HardwareBuffer allocateFrame(tc::ivec2 const size, std::uint32_t const format) {
    auto desc = AHardwareBuffer_Desc{};
    desc.width = static_cast<std::uint32_t>(size.x);
    desc.height = static_cast<std::uint32_t>(size.y);
    desc.layers = 1;
    desc.format = format;
    desc.usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN
               | AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;
    AHardwareBuffer *buffer = nullptr;
    if (AHardwareBuffer_allocate(&desc, &buffer) != 0) {
        throw tc::DeviceError{"AHardwareBuffer_allocate failed for format " + std::to_string(format)};
    }
    return HardwareBuffer{buffer, AHardwareBuffer_release};
}

std::uint8_t quantize(float const v) {
    return static_cast<std::uint8_t>(std::lround(std::fmin(std::fmax(v, 0.0f), 255.0f)));
}

// Writes the colour-bar pattern as BT.601 limited-range YUV 4:2:0 through
// the plane layout the allocator chose (NV12, NV21 or planar).
void fillYuv(AHardwareBuffer *buffer, tc::ivec2 const size) {
    auto planes = AHardwareBuffer_Planes{};
    if (AHardwareBuffer_lockPlanes(buffer, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, nullptr, &planes) != 0) {
        throw tc::DeviceError{"cannot lock the YUV frame for writing"};
    }
    auto const at = [](AHardwareBuffer_Plane const &plane, int const x, int const y) {
        return static_cast<std::uint8_t *>(plane.data) + static_cast<std::size_t>(y) * plane.rowStride
             + static_cast<std::size_t>(x) * plane.pixelStride;
    };
    for (int y = 0; y < size.y; ++y) {
        for (int x = 0; x < size.x; ++x) {
            auto const rgb = examples::testPattern(tc::ivec2{x, y}, size);
            *at(planes.planes[0], x, y) = quantize(16.0f + 219.0f * (0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z));
        }
    }
    for (int y = 0; y < size.y / 2; ++y) {
        for (int x = 0; x < size.x / 2; ++x) {
            auto const rgb = examples::testPattern(tc::ivec2{2 * x, 2 * y}, size);
            *at(planes.planes[1], x, y) = quantize(128.0f + 224.0f * (-0.168736f * rgb.x - 0.331264f * rgb.y + 0.5f * rgb.z));
            *at(planes.planes[2], x, y) = quantize(128.0f + 224.0f * (0.5f * rgb.x - 0.418688f * rgb.y - 0.081312f * rgb.z));
        }
    }
    AHardwareBuffer_unlock(buffer, nullptr);
}

void fillRgba(AHardwareBuffer *buffer, tc::ivec2 const size) {
    auto desc = AHardwareBuffer_Desc{};
    AHardwareBuffer_describe(buffer, &desc);
    void *data = nullptr;
    if (AHardwareBuffer_lock(buffer, AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN, -1, nullptr, &data) != 0) {
        throw tc::DeviceError{"cannot lock the RGBA frame for writing"};
    }
    for (int y = 0; y < size.y; ++y) {
        for (int x = 0; x < size.x; ++x) {
            auto const rgb = examples::testPattern(tc::ivec2{x, y}, size) * 255.0f;
            auto *texel = static_cast<std::uint8_t *>(data) + (static_cast<std::size_t>(y) * desc.stride + x) * 4;
            texel[0] = quantize(rgb.x);
            texel[1] = quantize(rgb.y);
            texel[2] = quantize(rgb.z);
            texel[3] = 255;
        }
    }
    AHardwareBuffer_unlock(buffer, nullptr);
}

struct Conversion {
    double milliseconds = 0.0;
    double meanError = 0.0;
    std::string barSamples; // RGB at the centre of each colour bar
};

Conversion convert(tc::Device &device, AHardwareBuffer *buffer, tc::ivec2 const size, std::string const &path) {
    auto frame = tc::ExternalImage::fromHardwareBuffer(buffer);
    auto rgba = device.createBuffer<tc::uint>(size.x * size.y);
    auto kernel = examples::ExternalFrameToRgba{};
    kernel.frame.attach(frame);
    kernel.rgba.attach(rgba);
    auto const start = std::chrono::steady_clock::now();
    device.uploadBuffer(rgba);
    device.useKernel(kernel);
    device.bindExternalTexture(kernel.frame);
    device.bindBuffer(kernel.rgba);
    device.execute(kernel, tc::uvec3{(static_cast<tc::uint>(size.x) + 15u) / 16u,
                                     (static_cast<tc::uint>(size.y) + 15u) / 16u, 1u});
    device.downloadBuffer(rgba);
    auto result = Conversion{};
    result.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

    auto image = examples::RgbaImage{size};
    auto total = 0.0;
    for (int y = 0; y < size.y; ++y) {
        for (int x = 0; x < size.x; ++x) {
            auto const packed = rgba[y * size.x + x];
            auto const expected = examples::testPattern(tc::ivec2{x, y}, size) * 255.0f;
            auto &pixel = image[tc::ivec2{x, y}];
            for (int c = 0; c < 4; ++c) {
                pixel.channels[c] = static_cast<std::uint8_t>((packed >> (8 * c)) & 0xffu);
            }
            for (int c = 0; c < 3; ++c) {
                total += std::abs(static_cast<double>(pixel.channels[c]) - static_cast<double>(expected[c]));
            }
        }
    }
    result.meanError = total / (3.0 * size.x * size.y);
    for (int bar = 0; bar < 8; ++bar) {
        auto const &pixel = image[tc::ivec2{(2 * bar + 1) * size.x / 16, size.y / 3}];
        result.barSamples += " (" + std::to_string(pixel.channels[0]) + "," + std::to_string(pixel.channels[1]) + ","
                           + std::to_string(pixel.channels[2]) + ")";
    }
    examples::writePpm(image, path);
    return result;
}

void report(char const *label, Conversion const &conversion) {
    std::cout << label << ": " << conversion.milliseconds << " ms, mean error " << conversion.meanError
              << " / 255, bars" << conversion.barSamples << "\n";
}

int run(examples::CommandLine const &line) {
    auto device = tc::createDevice(line.device);
    auto const info = device->info();
    std::cout << "backend " << info.backend << ": " << info.deviceName << "\n";
    auto const width = line.size > 0 ? line.size : 640;
    auto const size = tc::ivec2{width, width * 9 / 16} / 2 * 2;

    std::cout << "frames " << size.x << "x" << size.y << ", expected bars (255,255,255) (255,255,0) (0,255,255) "
              << "(0,255,0) (255,0,255) (255,0,0) (0,0,255) (0,0,0)\n";
    auto const yuv = allocateFrame(size, AHARDWAREBUFFER_FORMAT_Y8Cb8Cr8_420);
    fillYuv(yuv.get(), size);
    auto const rgbaFrame = allocateFrame(size, AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM);
    fillRgba(rgbaFrame.get(), size);
    // Two passes in alternating order separate format problems from
    // problems caused by an earlier conversion on the same device.
    report("YUV  (pass 1)", convert(*device, yuv.get(), size, "external_yuv_" + info.backend + ".ppm"));
    report("RGBA (pass 1)", convert(*device, rgbaFrame.get(), size, "external_rgba_" + info.backend + ".ppm"));
    report("RGBA (pass 2)", convert(*device, rgbaFrame.get(), size, "external_rgba_" + info.backend + ".ppm"));
    report("YUV  (pass 2)", convert(*device, yuv.get(), size, "external_yuv_" + info.backend + ".ppm"));
    return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
    auto const line = examples::parseCommandLine(argc, argv);
    try {
        return run(line);
    } catch (tc::DeviceError const &error) {
        std::cerr << "device error: " << error.what() << "\n";
        return 2;
    } catch (std::exception const &error) {
        std::cerr << "error: " << error.what() << "\n";
        return 3;
    }
}
