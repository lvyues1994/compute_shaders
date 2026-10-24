#include <tc/opengl/opengl_device.hpp>

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "egl_context.hpp"
#include "gl_api.hpp"
#include "gl_formats.hpp"

namespace tc::opengl {

namespace {

// Move-only owner of one OpenGL object name.
struct GlObject {
    using Deleter = void (*)(GlApi const &, GLuint);

    GlObject() = default;
    GlObject(GlApi const &api_, GLuint const id_, Deleter const deleter_) noexcept
        : api(&api_), id(id_), deleter(deleter_) {}
    GlObject(GlObject &&other) noexcept
        : api(other.api), id(std::exchange(other.id, 0u)), deleter(other.deleter) {}
    GlObject &operator=(GlObject &&other) noexcept {
        std::swap(api, other.api);
        std::swap(id, other.id);
        std::swap(deleter, other.deleter);
        return *this;
    }
    GlObject(GlObject const &) = delete;
    GlObject &operator=(GlObject const &) = delete;
    ~GlObject() {
        if (id != 0) {
            deleter(*api, id);
        }
    }

    GLuint get() const noexcept { return id; }

private:
    GlApi const *api = nullptr;
    GLuint id = 0;
    Deleter deleter = nullptr;
};

GlObject makeBuffer(GlApi const &gl) {
    auto id = GLuint{0};
    gl.glGenBuffers(1, &id);
    return GlObject{gl, id, [](GlApi const &api, GLuint name) { api.glDeleteBuffers(1, &name); }};
}

GlObject makeTexture(GlApi const &gl) {
    auto id = GLuint{0};
    gl.glGenTextures(1, &id);
    return GlObject{gl, id, [](GlApi const &api, GLuint name) { api.glDeleteTextures(1, &name); }};
}

GlObject compileProgram(GlApi const &gl, std::string const &source, std::string const &name) {
    auto const shader = gl.glCreateShader(GL_COMPUTE_SHADER);
    auto const *text = source.c_str();
    gl.glShaderSource(shader, 1, &text, nullptr);
    gl.glCompileShader(shader);
    auto status = GLint{0};
    gl.glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status != GL_TRUE) {
        char log[4096] = {};
        gl.glGetShaderInfoLog(shader, sizeof log, nullptr, log);
        gl.glDeleteShader(shader);
        throw DeviceError{"compiling kernel '" + name + "' failed:\n" + log};
    }
    auto program = GlObject{gl, gl.glCreateProgram(), [](GlApi const &api, GLuint id) { api.glDeleteProgram(id); }};
    gl.glAttachShader(program.get(), shader);
    gl.glLinkProgram(program.get());
    gl.glDeleteShader(shader);
    gl.glGetProgramiv(program.get(), GL_LINK_STATUS, &status);
    if (status != GL_TRUE) {
        char log[4096] = {};
        gl.glGetProgramInfoLog(program.get(), sizeof log, nullptr, log);
        throw DeviceError{"linking kernel '" + name + "' failed:\n" + log};
    }
    return program;
}

GLenum textureTarget(Dim const dim) {
    return dim == Dim::D3 ? GL_TEXTURE_3D : GL_TEXTURE_2D;
}

GLenum imageAccess(Access const access) {
    switch (access) {
    case Access::ReadOnly:
        return GL_READ_ONLY;
    case Access::WriteOnly:
        return GL_WRITE_ONLY;
    case Access::ReadWrite:
        return GL_READ_WRITE;
    }
    return GL_READ_WRITE;
}

struct BufferMirror {
    std::weak_ptr<void const> owner;
    GlObject buffer;
    std::size_t size = 0;
};

struct ImageMirror {
    std::weak_ptr<void const> owner;
    GlObject texture;
    ImageFormat format;
    Dim dim = Dim::D2;
    ivec3 extent{};
};

struct ProgramObject {
    KernelProgram reflection;
    GlObject program;
};

struct OpenGLDevice final : Device {
    explicit OpenGLDevice(OpenGLDeviceConfig const &config);

    DeviceInfo info() const override {
        return DeviceInfo{api == OpenGLApi::ES ? "gles" : "opengl", egl->renderer()};
    }

protected:
    void doUploadBuffer(ResourceIdentity const &identity, ConstByteSpan bytes) override;
    void doDownloadBuffer(ResourceIdentity const &identity, ByteSpan bytes) override;
    void doUploadImage(ImageTransfer const &transfer) override;
    void doDownloadImage(ImageTransfer const &transfer) override;
    void doUseKernel(KernelUse const &use) override;
    void doBindBuffer(BufferBindRequest const &request) override;
    void doBindImage(ImageBindRequest const &request) override;
    void doSetUniform(UniformWrite const &write) override;
    void doExecute(Dispatch const &dispatch) override;

private:
    GlApi const &gl() const noexcept { return egl->api(); }
    void collectGarbage();
    ProgramObject &programFor(std::string_view name);
    GLuint readbackProgram(ImageFormat const &format, Dim dim);
    void applyUniform(GLuint program, UniformWrite const &write, std::vector<std::byte> const &bytes) const;
    void checkLimits(KernelProgram const &program, uvec3 workGroups) const;

    std::unique_ptr<EglContext> egl; // declared first: destroyed after every GL object
    OpenGLApi api;
    std::unordered_map<void const *, BufferMirror> buffers;
    std::unordered_map<void const *, ImageMirror> images;
    std::unordered_map<std::string, ProgramObject> programs;
    std::map<std::string, GlObject> readbackPrograms;
    GlObject readbackBuffer;
    std::size_t readbackCapacity = 0;

    ProgramObject *current = nullptr;
    std::map<std::uint32_t, void const *> boundBuffers;
    std::map<std::uint32_t, void const *> boundImages;
    std::map<std::uint32_t, std::pair<UniformWrite, std::vector<std::byte>>> uniformValues;
};

OpenGLDevice::OpenGLDevice(OpenGLDeviceConfig const &config) : egl(EglContext::create(config)), api(config.api) {
    if (config.debugOutput and gl().glDebugMessageCallback != nullptr and gl().glEnable != nullptr) {
        gl().glEnable(GL_DEBUG_OUTPUT);
        gl().glDebugMessageCallback(
            [](GLenum, GLenum, GLuint, GLenum, GLsizei, GLchar const *message, void const *) {
                std::fprintf(stderr, "[gl] %s\n", message);
            },
            nullptr);
    }
    readbackBuffer = makeBuffer(gl());
    checkGl(gl(), "device setup");
}

// Drops device mirrors whose host resource no longer exists.
void OpenGLDevice::collectGarbage() {
    for (auto it = buffers.begin(); it != buffers.end();) {
        it = it->second.owner.expired() ? buffers.erase(it) : std::next(it);
    }
    for (auto it = images.begin(); it != images.end();) {
        it = it->second.owner.expired() ? images.erase(it) : std::next(it);
    }
}

void OpenGLDevice::doUploadBuffer(ResourceIdentity const &identity, ConstByteSpan const bytes) {
    egl->makeCurrent();
    collectGarbage();
    auto &mirror = buffers[identity.key()];
    if (mirror.buffer.get() == 0) {
        mirror.owner = identity.watch();
        mirror.buffer = makeBuffer(gl());
    }
    gl().glBindBuffer(GL_SHADER_STORAGE_BUFFER, mirror.buffer.get());
    if (mirror.size != bytes.size()) {
        gl().glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(bytes.size()), bytes.data(),
                          GL_DYNAMIC_COPY);
        mirror.size = bytes.size();
    } else {
        gl().glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(bytes.size()), bytes.data());
    }
    checkGl(gl(), "uploadBuffer");
}

void OpenGLDevice::doDownloadBuffer(ResourceIdentity const &identity, ByteSpan const bytes) {
    egl->makeCurrent();
    auto const it = buffers.find(identity.key());
    if (it == buffers.end() or it->second.size != bytes.size()) {
        throw DeviceError{"downloadBuffer: the buffer was never uploaded to this device (or changed size)"};
    }
    gl().glMemoryBarrier(GL_ALL_BARRIER_BITS);
    gl().glBindBuffer(GL_SHADER_STORAGE_BUFFER, it->second.buffer.get());
    auto const *mapped = gl().glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(bytes.size()),
                                               GL_MAP_READ_BIT);
    if (mapped == nullptr) {
        checkGl(gl(), "downloadBuffer");
        throw DeviceError{"downloadBuffer: glMapBufferRange failed"};
    }
    std::memcpy(bytes.data(), mapped, bytes.size());
    gl().glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    checkGl(gl(), "downloadBuffer");
}

void OpenGLDevice::doUploadImage(ImageTransfer const &transfer) {
    egl->makeCurrent();
    collectGarbage();
    auto const format = glFormatOf(transfer.format);
    auto const target = textureTarget(transfer.dim);
    auto &mirror = images[transfer.identity->key()];
    auto const sameShape = mirror.texture.get() != 0 and mirror.format == transfer.format
                       and mirror.extent == transfer.extent;
    if (not sameShape) {
        mirror.owner = transfer.identity->watch();
        mirror.texture = makeTexture(gl());
        mirror.format = transfer.format;
        mirror.dim = transfer.dim;
        mirror.extent = transfer.extent;
        gl().glBindTexture(target, mirror.texture.get());
        if (transfer.dim == Dim::D3) {
            gl().glTexStorage3D(target, 1, format.internalFormat, transfer.extent.x, transfer.extent.y,
                                transfer.extent.z);
        } else {
            gl().glTexStorage2D(target, 1, format.internalFormat, transfer.extent.x, transfer.extent.y);
        }
        gl().glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        gl().glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }
    auto const texelCount = static_cast<std::size_t>(transfer.extent.x) * transfer.extent.y * transfer.extent.z;
    auto texels = std::vector<std::byte>(texelCount * transfer.format.bytesPerTexel());
    transfer.writeTexels(ByteSpan{texels.data(), texels.size()});
    gl().glBindTexture(target, mirror.texture.get());
    gl().glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    if (transfer.dim == Dim::D3) {
        gl().glTexSubImage3D(target, 0, 0, 0, 0, transfer.extent.x, transfer.extent.y, transfer.extent.z,
                             format.pixelFormat, format.pixelType, texels.data());
    } else {
        gl().glTexSubImage2D(target, 0, 0, 0, transfer.extent.x, transfer.extent.y, format.pixelFormat,
                             format.pixelType, texels.data());
    }
    checkGl(gl(), "uploadImage");
}

GLuint OpenGLDevice::readbackProgram(ImageFormat const &format, Dim const dim) {
    auto const key = glslFormatQualifier(format) + (dim == Dim::D3 ? "3d" : "2d");
    auto it = readbackPrograms.find(key);
    if (it == readbackPrograms.end()) {
        auto program = compileProgram(gl(), readbackShaderSource(format, dim, api), "image readback " + key);
        it = readbackPrograms.emplace(key, std::move(program)).first;
    }
    return it->second.get();
}

void OpenGLDevice::doDownloadImage(ImageTransfer const &transfer) {
    egl->makeCurrent();
    auto const it = images.find(transfer.identity->key());
    if (it == images.end() or not(it->second.format == transfer.format) or not(it->second.extent == transfer.extent)) {
        throw DeviceError{"downloadImage: the image was never uploaded to this device"};
    }
    auto const &mirror = it->second;
    auto const format = glFormatOf(transfer.format);
    auto const texelCount = static_cast<std::size_t>(transfer.extent.x) * transfer.extent.y * transfer.extent.z;
    auto const readbackSize = texelCount * 4 * sizeof(std::uint32_t);
    gl().glBindBuffer(GL_SHADER_STORAGE_BUFFER, readbackBuffer.get());
    if (readbackCapacity < readbackSize) {
        gl().glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(readbackSize), nullptr, GL_STREAM_READ);
        readbackCapacity = readbackSize;
    }
    gl().glMemoryBarrier(GL_ALL_BARRIER_BITS);
    gl().glUseProgram(readbackProgram(transfer.format, transfer.dim));
    gl().glBindImageTexture(0, mirror.texture.get(), 0, transfer.dim == Dim::D3 ? GL_TRUE : GL_FALSE, 0,
                            GL_READ_ONLY, format.internalFormat);
    gl().glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, readbackBuffer.get());
    gl().glDispatchCompute((static_cast<GLuint>(transfer.extent.x) + 7u) / 8u,
                           (static_cast<GLuint>(transfer.extent.y) + 7u) / 8u,
                           static_cast<GLuint>(transfer.extent.z));
    gl().glMemoryBarrier(GL_ALL_BARRIER_BITS);
    auto const *mapped = static_cast<std::uint32_t const *>(gl().glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(readbackSize), GL_MAP_READ_BIT));
    if (mapped == nullptr) {
        checkGl(gl(), "downloadImage");
        throw DeviceError{"downloadImage: glMapBufferRange failed"};
    }
    auto const texelBytes = transfer.format.bytesPerTexel();
    auto texels = std::vector<std::byte>(texelCount * texelBytes);
    for (std::size_t i = 0; i < texelCount; ++i) {
        std::uint32_t const components[4] = {mapped[4 * i], mapped[4 * i + 1], mapped[4 * i + 2], mapped[4 * i + 3]};
        encodeTexel(transfer.format, components, texels.data() + i * texelBytes);
    }
    gl().glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    checkGl(gl(), "downloadImage");
    transfer.readTexels(ConstByteSpan{texels.data(), texels.size()});
}

ProgramObject &OpenGLDevice::programFor(std::string_view const name) {
    auto const key = std::string{name};
    if (auto const it = programs.find(key); it != programs.end()) {
        return it->second;
    }
    auto const reflection = ProgramRegistry::instance().find(name);
    if (not reflection) {
        throw DeviceError{"kernel '" + key + "' was not transpiled; add its header with tc_add_kernels()"};
    }
    auto const source = api == OpenGLApi::ES ? reflection->glslEs : reflection->glslDesktop;
    if (source.empty()) {
        throw DeviceError{"kernel '" + key + "' cannot run on OpenGL ES: " + std::string{reflection->esUnavailableReason}};
    }
    auto program = compileProgram(gl(), std::string{source}, key);
    return programs.emplace(key, ProgramObject{*reflection, std::move(program)}).first->second;
}

void OpenGLDevice::doUseKernel(KernelUse const &use) {
    egl->makeCurrent();
    auto &program = programFor(use.programName);
    if (not(program.reflection.localSize == use.localSize)) {
        throw DeviceError{"kernel '" + std::string{use.programName}
                          + "': local_size differs from the transpiled value; re-run the build"};
    }
    current = &program;
    boundBuffers.clear();
    boundImages.clear();
    uniformValues.clear();
}

void OpenGLDevice::doBindBuffer(BufferBindRequest const &request) {
    if (buffers.count(request.identity->key()) == 0) {
        throw DeviceError{"bindBuffer: binding " + std::to_string(request.binding)
                          + " refers to a buffer that was never uploaded"};
    }
    boundBuffers[request.binding] = request.identity->key();
}

void OpenGLDevice::doBindImage(ImageBindRequest const &request) {
    auto const it = images.find(request.identity->key());
    if (it == images.end() or not(it->second.format == request.format)) {
        throw DeviceError{"bindImage: binding " + std::to_string(request.binding)
                          + " refers to an image that was never uploaded"};
    }
    boundImages[request.binding] = request.identity->key();
}

void OpenGLDevice::doSetUniform(UniformWrite const &write) {
    auto bytes = std::vector<std::byte>(write.bytes.begin(), write.bytes.end());
    uniformValues[write.location] = {write, std::move(bytes)};
}

void OpenGLDevice::applyUniform(GLuint const program, UniformWrite const &write,
                                std::vector<std::byte> const &bytes) const {
    auto const location = static_cast<GLint>(write.location);
    auto const *data = bytes.data();
    auto const n = write.components;
    switch (write.scalar) {
    case ScalarKind::Int: {
        auto const *v = reinterpret_cast<GLint const *>(data);
        n == 1 ? gl().glProgramUniform1iv(program, location, 1, v)
        : n == 2 ? gl().glProgramUniform2iv(program, location, 1, v)
        : n == 3 ? gl().glProgramUniform3iv(program, location, 1, v)
                 : gl().glProgramUniform4iv(program, location, 1, v);
        break;
    }
    case ScalarKind::UInt: {
        auto const *v = reinterpret_cast<GLuint const *>(data);
        n == 1 ? gl().glProgramUniform1uiv(program, location, 1, v)
        : n == 2 ? gl().glProgramUniform2uiv(program, location, 1, v)
        : n == 3 ? gl().glProgramUniform3uiv(program, location, 1, v)
                 : gl().glProgramUniform4uiv(program, location, 1, v);
        break;
    }
    case ScalarKind::Float: {
        auto const *v = reinterpret_cast<GLfloat const *>(data);
        n == 1 ? gl().glProgramUniform1fv(program, location, 1, v)
        : n == 2 ? gl().glProgramUniform2fv(program, location, 1, v)
        : n == 3 ? gl().glProgramUniform3fv(program, location, 1, v)
                 : gl().glProgramUniform4fv(program, location, 1, v);
        break;
    }
    case ScalarKind::Double: {
        auto const setter = n == 1 ? gl().glProgramUniform1dv : n == 2 ? gl().glProgramUniform2dv
                          : n == 3 ? gl().glProgramUniform3dv
                                   : gl().glProgramUniform4dv;
        if (setter == nullptr) {
            throw DeviceError{"double uniforms need desktop OpenGL"};
        }
        setter(program, location, 1, reinterpret_cast<double const *>(data));
        break;
    }
    }
}

void OpenGLDevice::checkLimits(KernelProgram const &program, uvec3 const workGroups) const {
    auto invocations = GLint{0};
    gl().glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &invocations);
    auto const requested = program.localSize.x * program.localSize.y * program.localSize.z;
    if (requested > static_cast<uint>(invocations)) {
        throw DeviceError{"kernel '" + std::string{program.name} + "' uses " + std::to_string(requested)
                          + " invocations per workgroup; this device allows " + std::to_string(invocations)};
    }
    for (GLuint axis = 0; axis < 3; ++axis) {
        auto count = GLint{0};
        gl().glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, axis, &count);
        if (workGroups[static_cast<int>(axis)] > static_cast<uint>(count)) {
            throw DeviceError{"workgroup count exceeds the device limit of " + std::to_string(count)};
        }
    }
}

void OpenGLDevice::doExecute(Dispatch const &dispatch) {
    egl->makeCurrent();
    if (current == nullptr or current->reflection.name != dispatch.programName) {
        throw DeviceError{"execute: call useKernel() for '" + std::string{dispatch.programName} + "' first"};
    }
    auto const &reflection = current->reflection;
    checkLimits(reflection, dispatch.workGroups);
    auto const program = current->program.get();
    gl().glUseProgram(program);
    for (auto const &slot : reflection.resources) {
        auto const name = std::string{slot.name} + "' (binding " + std::to_string(slot.binding) + ")";
        if (slot.kind == ResourceKind::StorageBuffer) {
            auto const bound = boundBuffers.find(slot.binding);
            if (bound == boundBuffers.end()) {
                throw DeviceError{"execute: buffer '" + name + " is not bound"};
            }
            gl().glBindBufferBase(GL_SHADER_STORAGE_BUFFER, slot.binding, buffers.at(bound->second).buffer.get());
        } else {
            auto const bound = boundImages.find(slot.binding);
            if (bound == boundImages.end()) {
                throw DeviceError{"execute: image '" + name + " is not bound"};
            }
            auto const &mirror = images.at(bound->second);
            gl().glBindImageTexture(slot.binding, mirror.texture.get(), 0, mirror.dim == Dim::D3 ? GL_TRUE : GL_FALSE,
                                    0, imageAccess(slot.access), glFormatOf(mirror.format).internalFormat);
        }
    }
    for (auto const &slot : reflection.uniforms) {
        auto const value = uniformValues.find(slot.location);
        if (value == uniformValues.end()) {
            throw DeviceError{"execute: uniform '" + std::string{slot.name} + "' was never set with setUniform()"};
        }
        applyUniform(program, value->second.first, value->second.second);
    }
    gl().glDispatchCompute(dispatch.workGroups.x, dispatch.workGroups.y, dispatch.workGroups.z);
    gl().glMemoryBarrier(GL_ALL_BARRIER_BITS);
    checkGl(gl(), "execute");
}

} // namespace

std::unique_ptr<Device> createDevice(OpenGLDeviceConfig const &config) {
    return std::make_unique<OpenGLDevice>(config);
}

} // namespace tc::opengl
