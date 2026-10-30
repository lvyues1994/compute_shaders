#include <tc/external_image.hpp>

#include <cstring>
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

bool isRgba8(std::uint32_t const format) noexcept {
    return format == AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM or format == AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM;
}

// Copies between an RGBA8 or RGBA16F hardware buffer and the CPU frame.
void transferRgba(AHardwareBuffer *buffer, AHardwareBuffer_Desc const &desc, BufferResource<cpu::RGBA32F, Dim::D2> &frame,
                  bool const toFrame) {
    auto const usage = toFrame ? AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN : AHARDWAREBUFFER_USAGE_CPU_WRITE_OFTEN;
    void *data = nullptr;
    if (AHardwareBuffer_lock(buffer, usage, -1, nullptr, &data) != 0) {
        throw std::runtime_error{"cannot lock the hardware buffer for CPU access"};
    }
    auto const opaque = desc.format == AHARDWAREBUFFER_FORMAT_R8G8B8X8_UNORM;
    auto const texelBytes = isRgba8(desc.format) ? std::size_t{4} : std::size_t{8};
    for (std::uint32_t y = 0; y < desc.height; ++y) {
        for (std::uint32_t x = 0; x < desc.width; ++x) {
            auto *texel = static_cast<std::uint8_t *>(data) + (static_cast<std::size_t>(y) * desc.stride + x) * texelBytes;
            auto &pixel = frame[ivec2{static_cast<std::int32_t>(x), static_cast<std::int32_t>(y)}];
            if (isRgba8(desc.format)) {
                auto stored = cpu::RGBA8{{texel[0], texel[1], texel[2], texel[3]}};
                if (toFrame) {
                    pixel = convertPixel<cpu::RGBA32F>(stored);
                    pixel.channels[3] = opaque ? 1.0f : pixel.channels[3];
                } else {
                    stored = convertPixel<cpu::RGBA8>(pixel);
                    std::memcpy(texel, stored.channels, 4);
                }
            } else {
                auto stored = cpu::RGBA16F{};
                if (toFrame) {
                    std::memcpy(stored.channels, texel, 8);
                    pixel = convertPixel<cpu::RGBA32F>(stored);
                } else {
                    stored = convertPixel<cpu::RGBA16F>(pixel);
                    std::memcpy(texel, stored.channels, 8);
                }
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
    auto format = std::optional<ImageFormat>{};
    if (isRgba8(desc.format)) {
        format = GPUFormatTraits<gpu::RGBA8>::format;
    } else if (desc.format == AHARDWAREBUFFER_FORMAT_R16G16B16A16_FLOAT) {
        format = GPUFormatTraits<gpu::RGBA16F>::format;
    }
    return ExternalImage{std::move(native),
                         ivec2{static_cast<std::int32_t>(desc.width), static_cast<std::int32_t>(desc.height)}, format};
}

void ExternalImage::snapshot() {
    if (frame == nullptr) {
        frame = std::make_shared<BufferResource<cpu::RGBA32F, Dim::D2>>(extent);
    }
    auto desc = AHardwareBuffer_Desc{};
    AHardwareBuffer_describe(hardwareBuffer(), &desc);
    if (format) {
        transferRgba(hardwareBuffer(), desc, *frame, true);
    } else {
        snapshotYuv(hardwareBuffer(), desc, *frame);
    }
}

void ExternalImage::writeBack() {
    if (frame == nullptr or not format) {
        throw std::logic_error{"ExternalImage::writeBack: only RGBA frames bound as images can be written"};
    }
    auto desc = AHardwareBuffer_Desc{};
    AHardwareBuffer_describe(hardwareBuffer(), &desc);
    transferRgba(hardwareBuffer(), desc, *frame, false);
}

} // namespace tc

#else

namespace tc {

// External images come from platform buffers, which only exist on Android.
void ExternalImage::snapshot() {
    throw std::logic_error{"external images are only available on Android"};
}

void ExternalImage::writeBack() {
    throw std::logic_error{"external images are only available on Android"};
}

} // namespace tc

#endif
