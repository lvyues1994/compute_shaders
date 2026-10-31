#include "egl_context.hpp"

#include <EGL/eglext.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

namespace tc::opengl {

namespace {

bool hasExtension(char const *list, std::string_view const name) {
    if (list == nullptr) {
        return false;
    }
    auto const extensions = std::string_view{list};
    for (std::size_t pos = 0; pos < extensions.size();) {
        auto const end = std::min(extensions.find(' ', pos), extensions.size());
        if (extensions.substr(pos, end - pos) == name) {
            return true;
        }
        pos = end + 1;
    }
    return false;
}

struct DisplayCandidate {
    EGLDisplay display = EGL_NO_DISPLAY;
    bool isSoftware = false;
};

// One display per EGL device when the platform can enumerate devices (desktop
// Linux), then the default display (Android, or as a fallback).
std::vector<DisplayCandidate> displayCandidates() {
    auto candidates = std::vector<DisplayCandidate>{};
    auto const *clientExtensions = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
    if (hasExtension(clientExtensions, "EGL_EXT_device_enumeration")
        and hasExtension(clientExtensions, "EGL_EXT_platform_device")) {
        auto const queryDevices = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(eglGetProcAddress("eglQueryDevicesEXT"));
        auto const queryDeviceString =
            reinterpret_cast<PFNEGLQUERYDEVICESTRINGEXTPROC>(eglGetProcAddress("eglQueryDeviceStringEXT"));
        auto const getPlatformDisplay =
            reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(eglGetProcAddress("eglGetPlatformDisplayEXT"));
        auto count = EGLint{0};
        if (queryDevices != nullptr and queryDeviceString != nullptr and getPlatformDisplay != nullptr
            and queryDevices(0, nullptr, &count) == EGL_TRUE and count > 0) {
            auto devices = std::vector<EGLDeviceEXT>(static_cast<std::size_t>(count));
            queryDevices(count, devices.data(), &count);
            for (auto *const device : devices) {
                auto const isSoftware = hasExtension(queryDeviceString(device, EGL_EXTENSIONS), "EGL_MESA_device_software");
                auto *const display = getPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, device, nullptr);
                if (display != EGL_NO_DISPLAY) {
                    candidates.push_back(DisplayCandidate{display, isSoftware});
                }
            }
        }
    }
    if (candidates.empty()) {
        candidates.push_back(DisplayCandidate{eglGetDisplay(EGL_DEFAULT_DISPLAY), false});
    }
    std::stable_partition(candidates.begin(), candidates.end(),
                          [](DisplayCandidate const &c) { return not c.isSoftware; });
    return candidates;
}

std::vector<EGLint> contextAttributes(OpenGLDeviceConfig const &config) {
    auto attributes = std::vector<EGLint>{};
    if (config.api == OpenGLApi::Desktop) {
        attributes = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 3, EGL_CONTEXT_OPENGL_PROFILE_MASK,
                      EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT};
    } else {
        attributes = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1};
    }
    if (config.debugOutput) {
        attributes.push_back(EGL_CONTEXT_OPENGL_DEBUG);
        attributes.push_back(EGL_TRUE);
    }
    attributes.push_back(EGL_NONE);
    return attributes;
}

std::string apiName(OpenGLApi const api) {
    return api == OpenGLApi::Desktop ? "OpenGL 4.3 core" : "OpenGL ES 3.1";
}

struct DisplayUse {
    int leases = 0;
    bool terminates = true;
};

struct DisplayRegistry {
    std::mutex mutex;
    std::map<EGLDisplay, DisplayUse> uses;
};

DisplayRegistry &displayRegistry() {
    static auto registry = DisplayRegistry{};
    return registry;
}

} // namespace

std::string eglFailure(char const *const call) {
    auto stream = std::ostringstream{};
    stream << call << " failed (EGL error 0x" << std::hex << eglGetError() << ")";
    return stream.str();
}

std::optional<DisplayLease> DisplayLease::acquire(EGLDisplay const display) {
    auto &registry = displayRegistry();
    auto const lock = std::lock_guard<std::mutex>{registry.mutex};
    auto &use = registry.uses[display];
    if (use.leases == 0) {
        use.terminates = eglQueryString(display, EGL_VENDOR) == nullptr;
        auto major = EGLint{0};
        auto minor = EGLint{0};
        if (eglInitialize(display, &major, &minor) != EGL_TRUE) {
            registry.uses.erase(display);
            return std::nullopt;
        }
    }
    ++use.leases;
    return DisplayLease{display};
}

DisplayLease::DisplayLease(DisplayLease &&other) noexcept : display(std::exchange(other.display, EGL_NO_DISPLAY)) {}

DisplayLease::~DisplayLease() {
    if (display == EGL_NO_DISPLAY) {
        return;
    }
    auto &registry = displayRegistry();
    auto const lock = std::lock_guard<std::mutex>{registry.mutex};
    auto const use = registry.uses.find(display);
    if (--use->second.leases > 0) {
        return;
    }
    if (use->second.terminates) {
        eglTerminate(display);
    }
    registry.uses.erase(use);
}

std::unique_ptr<EglContext> EglContext::create(OpenGLDeviceConfig const &config) {
    auto tried = std::string{};
    for (auto const &candidate : displayCandidates()) {
        auto lease = DisplayLease::acquire(candidate.display);
        if (not lease) {
            continue;
        }
        auto egl = std::unique_ptr<EglContext>{new EglContext{}};
        egl->lease.emplace(std::move(*lease));
        egl->display = candidate.display;
        auto const api = config.api == OpenGLApi::Desktop ? EGL_OPENGL_API : EGL_OPENGL_ES_API;
        auto const renderable = config.api == OpenGLApi::Desktop ? EGL_OPENGL_BIT : EGL_OPENGL_ES3_BIT;
        if (eglBindAPI(api) != EGL_TRUE) {
            continue;
        }
        EGLint const pbufferConfig[] = {EGL_RENDERABLE_TYPE, renderable, EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_NONE};
        EGLint const anyConfig[] = {EGL_RENDERABLE_TYPE, renderable, EGL_NONE};
        auto eglConfig = EGLConfig{};
        auto configCount = EGLint{0};
        if (eglChooseConfig(egl->display, pbufferConfig, &eglConfig, 1, &configCount) != EGL_TRUE or configCount == 0) {
            if (eglChooseConfig(egl->display, anyConfig, &eglConfig, 1, &configCount) != EGL_TRUE or configCount == 0) {
                continue;
            }
        }
        auto const attributes = contextAttributes(config);
        egl->context = eglCreateContext(egl->display, eglConfig, EGL_NO_CONTEXT, attributes.data());
        if (egl->context == EGL_NO_CONTEXT) {
            continue;
        }
        if (not hasExtension(eglQueryString(egl->display, EGL_EXTENSIONS), "EGL_KHR_surfaceless_context")) {
            EGLint const size[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
            egl->surface = eglCreatePbufferSurface(egl->display, eglConfig, size);
        }
        if (eglMakeCurrent(egl->display, egl->surface, egl->surface, egl->context) != EGL_TRUE) {
            continue;
        }
        try {
            egl->gl = GlApi::load();
        } catch (DeviceError const &error) {
            tried += std::string{" ("} + error.what() + ")";
            continue;
        }
        egl->rendererName = reinterpret_cast<char const *>(egl->gl.glGetString(GL_RENDERER));
        auto const *displayExtensions = eglQueryString(egl->display, EGL_EXTENSIONS);
        auto &sync = egl->syncApi;
        sync.create = reinterpret_cast<PFNEGLCREATESYNCKHRPROC>(eglGetProcAddress("eglCreateSyncKHR"));
        sync.destroy = reinterpret_cast<PFNEGLDESTROYSYNCKHRPROC>(eglGetProcAddress("eglDestroySyncKHR"));
        sync.clientWait = reinterpret_cast<PFNEGLCLIENTWAITSYNCKHRPROC>(eglGetProcAddress("eglClientWaitSyncKHR"));
        sync.getAttribute = reinterpret_cast<PFNEGLGETSYNCATTRIBKHRPROC>(eglGetProcAddress("eglGetSyncAttribKHR"));
        sync.serverWait = reinterpret_cast<PFNEGLWAITSYNCKHRPROC>(eglGetProcAddress("eglWaitSyncKHR"));
        sync.dupNativeFence =
            reinterpret_cast<PFNEGLDUPNATIVEFENCEFDANDROIDPROC>(eglGetProcAddress("eglDupNativeFenceFDANDROID"));
        sync.fences = hasExtension(displayExtensions, "EGL_KHR_fence_sync") and sync.create != nullptr
                  and sync.destroy != nullptr and sync.clientWait != nullptr and sync.getAttribute != nullptr;
        sync.serverWaits = sync.fences and hasExtension(displayExtensions, "EGL_KHR_wait_sync") and sync.serverWait != nullptr;
        sync.nativeFences = sync.fences and hasExtension(displayExtensions, "EGL_ANDROID_native_fence_sync")
                        and sync.dupNativeFence != nullptr;
        if (not config.deviceNameContains.empty()
            and egl->rendererName.find(config.deviceNameContains) == std::string::npos) {
            tried += " '" + egl->rendererName + "'";
            continue;
        }
        return egl;
    }
    throw DeviceError{"no EGL device provides " + apiName(config.api)
                      + (config.deviceNameContains.empty() ? std::string{}
                                                           : " matching '" + config.deviceNameContains + "'")
                      + (tried.empty() ? std::string{} : " (found" + tried + ")")};
}

EglContext::~EglContext() {
    if (display == EGL_NO_DISPLAY) {
        return;
    }
    if (context != EGL_NO_CONTEXT) {
        eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglDestroyContext(display, context);
    }
    if (surface != EGL_NO_SURFACE) {
        eglDestroySurface(display, surface);
    }
}

void EglContext::makeCurrent() const {
    if (eglGetCurrentContext() != context
        and eglMakeCurrent(display, surface, surface, context) != EGL_TRUE) {
        throw DeviceError{eglFailure("eglMakeCurrent")};
    }
}

} // namespace tc::opengl
