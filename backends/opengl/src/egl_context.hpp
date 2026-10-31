#pragma once

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <memory>
#include <optional>
#include <string>

#include <tc/opengl/opengl_device.hpp>

#include "gl_api.hpp"

namespace tc::opengl {

// EGL sync objects: fences any thread can wait on without a current
// context (EGL_KHR_fence_sync), GPU-side waits (EGL_KHR_wait_sync) and
// Android sync file descriptors (EGL_ANDROID_native_fence_sync).
struct EglSyncApi {
    PFNEGLCREATESYNCKHRPROC create = nullptr;
    PFNEGLDESTROYSYNCKHRPROC destroy = nullptr;
    PFNEGLCLIENTWAITSYNCKHRPROC clientWait = nullptr;
    PFNEGLGETSYNCATTRIBKHRPROC getAttribute = nullptr;
    PFNEGLWAITSYNCKHRPROC serverWait = nullptr;
    PFNEGLDUPNATIVEFENCEFDANDROIDPROC dupNativeFence = nullptr;
    bool fences = false;
    bool serverWaits = false;
    bool nativeFences = false;
};

// "`call` failed (EGL error 0x....)", with the calling thread's eglGetError().
std::string eglFailure(char const *call);

// One device's hold on an initialized display. eglInitialize does not count, and one
// eglTerminate ends the display for every context on it, so the leases on a display
// share one initialization: the last one released terminates it, unless the display
// was already initialized when tc first leased it.
class DisplayLease {
public:
    // Empty when the display cannot be initialized.
    static std::optional<DisplayLease> acquire(EGLDisplay display);

    DisplayLease(DisplayLease &&other) noexcept;
    DisplayLease(DisplayLease const &) = delete;
    DisplayLease &operator=(DisplayLease const &) = delete;
    DisplayLease &operator=(DisplayLease &&) = delete;
    ~DisplayLease();

    EGLDisplay get() const noexcept { return display; }

private:
    explicit DisplayLease(EGLDisplay display_) noexcept : display(display_) {}

    EGLDisplay display;
};

// A headless EGL display + context made current on the creating thread,
// together with the GL entry points loaded for it.
struct EglContext {
    // Tries every EGL device (hardware first, software renderers last) and
    // keeps the first whose GL_RENDERER contains `config.deviceNameContains`.
    static std::unique_ptr<EglContext> create(OpenGLDeviceConfig const &config);

    EglContext(EglContext const &) = delete;
    EglContext &operator=(EglContext const &) = delete;
    ~EglContext();

    void makeCurrent() const;
    EGLDisplay eglDisplay() const noexcept { return display; }
    EglSyncApi const &sync() const noexcept { return syncApi; }
    GlApi const &api() const noexcept { return gl; }
    std::string const &renderer() const noexcept { return rendererName; }

private:
    EglContext() = default;

    std::optional<DisplayLease> lease; // ends after the context and surface
    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE;
    GlApi gl;
    EglSyncApi syncApi;
    std::string rendererName;
};

} // namespace tc::opengl
