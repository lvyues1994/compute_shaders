#include <tc/external_image.hpp>

#include <stdexcept>

#if defined(__ANDROID__)

namespace tc {

namespace {

// ITU-R BT.601 limited range, the same model the Vulkan backend asks its
// sampler YCbCr conversion for.
vec4 bt601LimitedToRgb(float const y, float const cb, float const cr) noexcept {
    auto const luma = (y - 16.0f) / 219.0f;
    auto const blue = (cb - 128.0f) / 224.0f;
    auto const red = (cr - 128.0f) / 224.0f;
    return vec4{luma + 1.402f * red, luma - 0.344136f * blue - 0.714136f * red, luma + 1.772f * blue, 1.0f};
}

void snapshotRgba(AHardwareBuffer *buffer, AHardwareBuffer_Desc const &desc, BufferResource<cpu::RGBA32F, Dim::D2> &frame) {
    void *data = nullptr;
    if (AHardwareBuffer_lock(buffer, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, -1, nullptr, &data) != 0) {
        throw std::runtime_error{"cannot lock the hardware buffer for CPU reading"};
    }
    auto const *bytes = static_cast<std::uint8_t const *>(data);
    auto const opaque = desc.format == AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM;
    for (std::uint32_t y = 0; y < desc.height; ++y) {
        for (std::uint32_t x = 0; x < desc.width; ++x) {
            auto const *texel = bytes + (static_cast<std::size_t>(y) * desc.stride + x) * 4;
            auto &pixel = frame[ivec2{static_cast<std::int32_t>(x), static_cast<std::int32_t>(y)}];
            for (int c = 0; c < 4; ++c) {
                pixel.channels[c] = static_cast<float>(texel[c]) / 255.0f;
            }
            if (opaque) {
                pixel.channels[3] = 1.0f;
            }
        }
    }
    AHardwareBuffer_unlock(buffer, nullptr);
}

void snapshotYuv(AHardwareBuffer *buffer, AHardwareBuffer_Desc const &desc, BufferResource<cpu::RGBA32F, Dim::D2> &frame) {
    auto planes = AHardwareBuffer_Planes{};
    if (AHardwareBuffer_lockPlanes(buffer, AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN, -1, nullptr, &planes) != 0
        or planes.planeCount < 3) {
        throw std::runtime_error{"cannot lock the YUV planes of the hardware buffer for CPU reading"};
    }
    auto const sample = [](AHardwareBuffer_Plane const &plane, std::uint32_t const x, std::uint32_t const y) {
        auto const *bytes = static_cast<std::uint8_t const *>(plane.data);
        return static_cast<float>(bytes[static_cast<std::size_t>(y) * plane.rowStride + x * plane.pixelStride]);
    };
    for (std::uint32_t y = 0; y < desc.height; ++y) {
        for (std::uint32_t x = 0; x < desc.width; ++x) {
            auto const rgb = bt601LimitedToRgb(sample(planes.planes[0], x, y), sample(planes.planes[1], x / 2, y / 2),
                                               sample(planes.planes[2], x / 2, y / 2));
            auto &pixel = frame[ivec2{static_cast<std::int32_t>(x), static_cast<std::int32_t>(y)}];
            for (int c = 0; c < 4; ++c) {
                pixel.channels[c] = rgb[c];
            }
        }
    }
    AHardwareBuffer_unlock(buffer, nullptr);
}

} // namespace

ExternalImage ExternalImage::fromHardwareBuffer(AHardwareBuffer *buffer) {
    if (buffer == nullptr) {
        throw std::invalid_argument{"ExternalImage::fromHardwareBuffer: null buffer"};
    }
    AHardwareBuffer_acquire(buffer);
    auto native = std::shared_ptr<void>{buffer, [](void *b) { AHardwareBuffer_release(static_cast<AHardwareBuffer *>(b)); }};
    auto desc = AHardwareBuffer_Desc{};
    AHardwareBuffer_describe(buffer, &desc);
    return ExternalImage{std::move(native),
                         ivec2{static_cast<std::int32_t>(desc.width), static_cast<std::int32_t>(desc.height)}};
}

void ExternalImage::snapshot() {
    if (frame == nullptr) {
        frame = std::make_shared<BufferResource<cpu::RGBA32F, Dim::D2>>(extent);
    }
    auto desc = AHardwareBuffer_Desc{};
    AHardwareBuffer_describe(hardwareBuffer(), &desc);
    if (desc.format == AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM or desc.format == AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM) {
        snapshotRgba(hardwareBuffer(), desc, *frame);
    } else {
        snapshotYuv(hardwareBuffer(), desc, *frame);
    }
}

} // namespace tc

#else

namespace tc {

// External images come from platform buffers, which only exist on Android.
void ExternalImage::snapshot() {
    throw std::logic_error{"external images are only available on Android"};
}

} // namespace tc

#endif
