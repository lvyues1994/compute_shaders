#include "gl_api.hpp"

#include <dlfcn.h>

#include <EGL/egl.h>

#include <string>

#include <tc/device.hpp>

namespace tc::opengl {

namespace {

// eglGetProcAddress may not return core entry points on older Android EGL
// implementations; those are then exported by the linked libGLESv3.
void *lookup(char const *name) {
    if (auto const address = eglGetProcAddress(name)) {
        return reinterpret_cast<void *>(address);
    }
    return dlsym(RTLD_DEFAULT, name);
}

} // namespace

GlApi GlApi::load() {
    auto api = GlApi{};
#define TC_GL_LOAD_REQUIRED(type, name)                                                                                \
    api.name = reinterpret_cast<type>(lookup(#name));                                                                  \
    if (api.name == nullptr) {                                                                                         \
        throw DeviceError{"OpenGL entry point " #name " is unavailable"};                                             \
    }
#define TC_GL_LOAD_OPTIONAL(type, name) api.name = reinterpret_cast<type>(lookup(#name));
    TC_GL_REQUIRED_FUNCTIONS(TC_GL_LOAD_REQUIRED)
    TC_GL_OPTIONAL_FUNCTIONS(TC_GL_LOAD_OPTIONAL)
#undef TC_GL_LOAD_REQUIRED
#undef TC_GL_LOAD_OPTIONAL
    return api;
}

void checkGl(GlApi const &gl, char const *operation) {
    auto const error = gl.glGetError();
    if (error == GL_NO_ERROR) {
        return;
    }
    while (gl.glGetError() != GL_NO_ERROR) {
    }
    throw DeviceError{std::string{operation} + " failed with OpenGL error " + std::to_string(error)};
}

} // namespace tc::opengl
