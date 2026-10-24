#include "external_texture.hpp"

#include <tc/device.hpp>

#if defined(__ANDROID__)

#include <EGL/eglext.h>
#include <android/hardware_buffer.h>

namespace tc::opengl {

namespace {

using PfnEglImageTargetTexture2D = void(GL_APIENTRY *)(GLenum, void *);

struct AndroidExternalTexture final : ExternalTexture {
    AndroidExternalTexture(EglContext const &egl_, ExternalImage const &image) : egl(egl_) {
        auto const getClientBuffer = reinterpret_cast<PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC>(
            eglGetProcAddress("eglGetNativeClientBufferANDROID"));
        auto const createImage = reinterpret_cast<PFNEGLCREATEIMAGEKHRPROC>(eglGetProcAddress("eglCreateImageKHR"));
        destroyImage = reinterpret_cast<PFNEGLDESTROYIMAGEKHRPROC>(eglGetProcAddress("eglDestroyImageKHR"));
        auto const targetTexture =
            reinterpret_cast<PfnEglImageTargetTexture2D>(eglGetProcAddress("glEGLImageTargetTexture2DOES"));
        if (getClientBuffer == nullptr or createImage == nullptr or destroyImage == nullptr or targetTexture == nullptr) {
            throw DeviceError{"importing hardware buffers needs EGL_ANDROID_get_native_client_buffer and "
                              "GL_OES_EGL_image_external"};
        }
        auto description = AHardwareBuffer_Desc{};
        AHardwareBuffer_describe(image.hardwareBuffer(), &description);
        pixelFormat = description.format;
        EGLint const attributes[] = {EGL_IMAGE_PRESERVED_KHR, EGL_TRUE, EGL_NONE};
        eglImage = createImage(egl.eglDisplay(), EGL_NO_CONTEXT, EGL_NATIVE_BUFFER_ANDROID,
                               getClientBuffer(image.hardwareBuffer()), attributes);
        if (eglImage == EGL_NO_IMAGE_KHR) {
            throw DeviceError{"eglCreateImageKHR failed for the hardware buffer"};
        }
        auto const &gl = egl.api();
        gl.glGenTextures(1, &name);
        gl.glBindTexture(kTextureExternalOes, name);
        targetTexture(kTextureExternalOes, eglImage);
        gl.glTexParameteri(kTextureExternalOes, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        gl.glTexParameteri(kTextureExternalOes, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        gl.glTexParameteri(kTextureExternalOes, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        gl.glTexParameteri(kTextureExternalOes, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        checkGl(gl, "importing a hardware buffer");
    }

    AndroidExternalTexture(AndroidExternalTexture const &) = delete;
    AndroidExternalTexture &operator=(AndroidExternalTexture const &) = delete;

    ~AndroidExternalTexture() override {
        egl.api().glDeleteTextures(1, &name);
        destroyImage(egl.eglDisplay(), eglImage);
    }

    GLuint texture() const noexcept override { return name; }
    std::uint32_t format() const noexcept override { return pixelFormat; }

private:
    EglContext const &egl; // the EGL image holds its own reference to the hardware buffer
    PFNEGLDESTROYIMAGEKHRPROC destroyImage = nullptr;
    EGLImageKHR eglImage = EGL_NO_IMAGE_KHR;
    GLuint name = 0;
    std::uint32_t pixelFormat = 0;
};

} // namespace

std::unique_ptr<ExternalTexture> importExternalTexture(EglContext const &egl, ExternalImage const &image) {
    return std::make_unique<AndroidExternalTexture>(egl, image);
}

} // namespace tc::opengl

#else

namespace tc::opengl {

std::unique_ptr<ExternalTexture> importExternalTexture(EglContext const &, ExternalImage const &) {
    throw DeviceError{"external images are only available on Android"};
}

} // namespace tc::opengl

#endif
