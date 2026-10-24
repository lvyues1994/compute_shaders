#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include <examples/ppm.hpp>
#include <tc/device.hpp>

#include "nv12_to_rgba.hpp"

namespace examples {

struct Nv12Frame {
    tc::BufferResource<tc::cpu::R8, tc::Dim::D2> luma;
    tc::BufferResource<tc::cpu::RG8, tc::Dim::D2> chroma;
};

// YUV -> RGB as `toRgb * (yuv - offset)` on normalized [0, 1] values.
struct YuvStandard {
    tc::mat3 toRgb;
    tc::vec3 offset;
};

// ITU-R BT.601, limited ("video") range: Y in [16, 235], UV in [16, 240].
inline YuvStandard bt601Limited() {
    auto const y = 255.0f / 219.0f;
    auto const c = 255.0f / 224.0f;
    auto standard = YuvStandard{};
    standard.toRgb = tc::mat3{tc::vec3{y, y, y},                          // Y column
                              tc::vec3{0.0f, -0.344136f * c, 1.772f * c}, // U (Cb) column
                              tc::vec3{1.402f * c, -0.714136f * c, 0.0f}}; // V (Cr) column
    standard.offset = tc::vec3{16.0f / 255.0f, 128.0f / 255.0f, 128.0f / 255.0f};
    return standard;
}

// The RGB colour of the synthetic test frame at pixel (x, y): eight colour
// bars over a grey ramp.
inline tc::vec3 testPattern(tc::ivec2 const p, tc::ivec2 const size) {
    static constexpr std::array<tc::vec3, 8> bars{
        tc::vec3{1.0f, 1.0f, 1.0f}, tc::vec3{1.0f, 1.0f, 0.0f}, tc::vec3{0.0f, 1.0f, 1.0f},
        tc::vec3{0.0f, 1.0f, 0.0f}, tc::vec3{1.0f, 0.0f, 1.0f}, tc::vec3{1.0f, 0.0f, 0.0f},
        tc::vec3{0.0f, 0.0f, 1.0f}, tc::vec3{0.0f, 0.0f, 0.0f}};
    if (p.y < size.y * 2 / 3) {
        return bars[static_cast<std::size_t>(p.x * 8 / size.x)];
    }
    return tc::vec3{static_cast<float>(p.x) / static_cast<float>(size.x - 1)};
}

// Encodes the test pattern as BT.601 limited-range NV12 on the host.
inline Nv12Frame makeTestFrame(tc::ivec2 const size) {
    auto frame = Nv12Frame{tc::BufferResource<tc::cpu::R8, tc::Dim::D2>{size},
                           tc::BufferResource<tc::cpu::RG8, tc::Dim::D2>{size / 2}};
    auto const quantize = [](float const v) {
        return static_cast<std::uint8_t>(std::clamp(std::lround(v), 0L, 255L));
    };
    for (int y = 0; y < size.y; ++y) {
        for (int x = 0; x < size.x; ++x) {
            auto const rgb = testPattern(tc::ivec2{x, y}, size);
            auto const luma = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
            frame.luma[tc::ivec2{x, y}].channels[0] = quantize(16.0f + 219.0f * luma);
        }
    }
    for (int y = 0; y < size.y / 2; ++y) {
        for (int x = 0; x < size.x / 2; ++x) {
            auto rgb = tc::vec3{0.0f};
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    rgb += testPattern(tc::ivec2{2 * x + dx, 2 * y + dy}, size) * 0.25f;
                }
            }
            auto const cb = -0.168736f * rgb.x - 0.331264f * rgb.y + 0.5f * rgb.z;
            auto const cr = 0.5f * rgb.x - 0.418688f * rgb.y - 0.081312f * rgb.z;
            auto &uv = frame.chroma[tc::ivec2{x, y}];
            uv.channels[0] = quantize(128.0f + 224.0f * cb);
            uv.channels[1] = quantize(128.0f + 224.0f * cr);
        }
    }
    return frame;
}

inline RgbaImage convertNv12(tc::Device &device, Nv12Frame &frame, YuvStandard const &standard) {
    auto rgba = RgbaImage{frame.luma.extent()};
    auto kernel = Nv12ToRgba{};
    kernel.luma.attach(frame.luma);
    kernel.chroma.attach(frame.chroma);
    kernel.rgba.attach(rgba);
    kernel.yuvToRgb = standard.toRgb;
    kernel.yuvOffset = standard.offset;
    device.uploadImage(kernel.luma);
    device.uploadImage(kernel.chroma);
    device.uploadImage(kernel.rgba);
    device.useKernel(kernel);
    device.bindTexture(kernel.luma);
    device.bindTexture(kernel.chroma);
    device.bindImage(kernel.rgba);
    device.setUniform(kernel.yuvToRgb);
    device.setUniform(kernel.yuvOffset);
    auto const size = rgba.extent();
    device.execute(kernel, tc::uvec3{(static_cast<tc::uint>(size.x) + 15u) / 16u,
                                     (static_cast<tc::uint>(size.y) + 15u) / 16u, 1u});
    device.downloadImage(kernel.rgba);
    return rgba;
}

} // namespace examples
