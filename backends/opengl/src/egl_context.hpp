#pragma once

#include <EGL/egl.h>

#include <memory>
#include <string>

#include <tc/opengl/opengl_device.hpp>

#include "gl_api.hpp"

namespace tc::opengl {

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
    GlApi const &api() const noexcept { return gl; }
    std::string const &renderer() const noexcept { return rendererName; }

private:
    EglContext() = default;

    EGLDisplay display = EGL_NO_DISPLAY;
    EGLContext context = EGL_NO_CONTEXT;
    EGLSurface surface = EGL_NO_SURFACE;
    GlApi gl;
    std::string rendererName;
};

} // namespace tc::opengl
