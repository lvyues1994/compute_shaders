#include <tc/opengl/opengl_device.hpp>

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "egl_context.hpp"
#include "external_texture.hpp"
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

// A persistently mapped buffer from createBuffer(). It may outlive the
// device, so it holds the EGL context; it must be released on the GL thread.
struct MappedBuffer {
    std::shared_ptr<EglContext> egl;
    GLuint buffer = 0;
    std::byte *mapped = nullptr;

    MappedBuffer() = default;
    MappedBuffer(MappedBuffer const &) = delete;
    MappedBuffer &operator=(MappedBuffer const &) = delete;
    ~MappedBuffer() {
        if (buffer == 0) {
            return;
        }
        try {
            egl->makeCurrent();
        } catch (DeviceError const &) {
            return; // context unavailable on this thread: the driver frees the buffer with the context
        }
        auto const &gl = egl->api();
        gl.glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        gl.glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        gl.glDeleteBuffers(1, &buffer);
    }
};

struct BufferMirror {
    std::weak_ptr<void const> owner;
    GlObject buffer;
    std::shared_ptr<MappedBuffer> mapped; // set for createBuffer() allocations instead of `buffer`
    std::size_t size = 0;

    GLuint handle() const noexcept { return mapped ? mapped->buffer : buffer.get(); }
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
    // Adreno drivers fix the colour conversion of a samplerExternalOES at the
    // first dispatch of a program, so each combination of external texture
    // formats gets its own link of the same source.
    std::map<std::vector<std::uint32_t>, GlObject> externalVariants;
};

struct BoundTexture {
    void const *key = nullptr;
    Filter filter = Filter::Nearest;
    Wrap wrap = Wrap::ClampToEdge;
};

struct ExternalMirror {
    std::weak_ptr<void const> owner;
    std::unique_ptr<ExternalTexture> texture;
};

struct OpenGLDevice final : Device {
    explicit OpenGLDevice(OpenGLDeviceConfig const &config);

    DeviceInfo info() const override {
        return DeviceInfo{api == OpenGLApi::ES ? "gles" : "opengl", egl->renderer()};
    }

protected:
    std::optional<ExternalMemory> doAllocateBuffer(ResourceIdentity const &identity, std::size_t bytes) override;
    void doUploadBuffer(ResourceIdentity const &identity, ConstByteSpan bytes) override;
    void doDownloadBuffer(ResourceIdentity const &identity, ByteSpan bytes) override;
    void doUploadImage(ImageTransfer const &transfer) override;
    void doDownloadImage(ImageTransfer const &transfer) override;
    void doUseKernel(KernelUse const &use) override;
    void doBindBuffer(BufferBindRequest const &request) override;
    void doBindImage(ImageBindRequest const &request) override;
    void doBindTexture(TextureBindRequest const &request) override;
    void doBindExternalTexture(ExternalTextureBindRequest const &request) override;
    void doSetUniform(UniformWrite const &write) override;
    void doExecute(Dispatch const &dispatch) override;
    void doFinish() override {
        egl->makeCurrent();
        gl().glMemoryBarrier(GL_ALL_BARRIER_BITS);
        gl().glFinish();
    }

private:
    GlApi const &gl() const noexcept { return egl->api(); }
    PfnBufferStorage bufferStorage() const;
    GLuint samplerFor(Filter filter, Wrap wrap);
    void bindTextureUnit(GLuint unit, ImageMirror const &mirror, Filter filter, Wrap wrap);
    void collectGarbage();
    ProgramObject &programFor(std::string_view name);
    GLuint programForBoundExternals(ProgramObject &program);
    GLuint readbackProgram(ImageFormat const &format, Dim dim);
    void applyUniform(GLuint program, UniformWrite const &write, std::vector<std::byte> const &bytes) const;
    void checkLimits(KernelProgram const &program, uvec3 workGroups) const;

    std::shared_ptr<EglContext> egl; // declared first: destroyed after every GL object
    OpenGLApi api;
    std::unordered_map<void const *, BufferMirror> buffers;
    std::unordered_map<void const *, ImageMirror> images;
    std::unordered_map<void const *, ExternalMirror> externals;
    std::unordered_map<std::string, ProgramObject> programs;
    std::map<std::string, GlObject> readbackPrograms;
    std::map<std::pair<Filter, Wrap>, GlObject> samplers;
    GlObject readbackBuffer;
    std::size_t readbackCapacity = 0;

    ProgramObject *current = nullptr;
    std::map<std::uint32_t, void const *> boundBuffers;
    std::map<std::uint32_t, void const *> boundImages;
    std::map<std::uint32_t, BoundTexture> boundTextures;
    std::map<std::uint32_t, void const *> boundExternals;
    std::map<std::uint32_t, std::pair<UniformWrite, std::vector<std::byte>>> uniformValues;
};

GLenum wrapMode(Wrap const wrap) {
    switch (wrap) {
    case Wrap::ClampToEdge:
        return GL_CLAMP_TO_EDGE;
    case Wrap::Repeat:
        return GL_REPEAT;
    case Wrap::MirroredRepeat:
        return GL_MIRRORED_REPEAT;
    }
    return GL_CLAMP_TO_EDGE;
}

bool hasGlExtension(GlApi const &gl, char const *name) {
    auto count = GLint{0};
    gl.glGetIntegerv(GL_NUM_EXTENSIONS, &count);
    for (GLint i = 0; i < count; ++i) {
        auto const *extension = reinterpret_cast<char const *>(gl.glGetStringi(GL_EXTENSIONS, static_cast<GLuint>(i)));
        if (extension != nullptr and std::strcmp(extension, name) == 0) {
            return true;
        }
    }
    return false;
}

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
    for (auto it = externals.begin(); it != externals.end();) {
        it = it->second.owner.expired() ? externals.erase(it) : std::next(it);
    }
}

PfnBufferStorage OpenGLDevice::bufferStorage() const {
    if (api == OpenGLApi::ES) {
        return hasGlExtension(gl(), "GL_EXT_buffer_storage") ? gl().glBufferStorageEXT : nullptr;
    }
    auto major = GLint{0};
    auto minor = GLint{0};
    gl().glGetIntegerv(GL_MAJOR_VERSION, &major);
    gl().glGetIntegerv(GL_MINOR_VERSION, &minor);
    auto const core44 = major > 4 or (major == 4 and minor >= 4);
    return core44 or hasGlExtension(gl(), "GL_ARB_buffer_storage") ? gl().glBufferStorage : nullptr;
}

std::optional<ExternalMemory> OpenGLDevice::doAllocateBuffer(ResourceIdentity const &identity,
                                                             std::size_t const bytes) {
    egl->makeCurrent();
    auto const storage = bufferStorage();
    if (storage == nullptr or bytes == 0) {
        return std::nullopt;
    }
    collectGarbage();
    auto allocation = std::make_shared<MappedBuffer>();
    allocation->egl = egl;
    gl().glGenBuffers(1, &allocation->buffer);
    gl().glBindBuffer(GL_SHADER_STORAGE_BUFFER, allocation->buffer);
    auto const flags = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | kMapPersistentBit | kMapCoherentBit;
    storage(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(bytes), nullptr, flags);
    allocation->mapped = static_cast<std::byte *>(
        gl().glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, static_cast<GLsizeiptr>(bytes), flags));
    if (allocation->mapped == nullptr) {
        checkGl(gl(), "createBuffer");
        throw DeviceError{"createBuffer: persistent glMapBufferRange failed"};
    }
    std::memset(allocation->mapped, 0, bytes);
    checkGl(gl(), "createBuffer");

    auto &mirror = buffers[identity.key()];
    mirror = BufferMirror{};
    mirror.owner = identity.watch();
    mirror.mapped = allocation;
    mirror.size = bytes;
    return ExternalMemory{allocation->mapped, bytes, allocation};
}

void OpenGLDevice::doUploadBuffer(ResourceIdentity const &identity, ConstByteSpan const bytes) {
    egl->makeCurrent();
    collectGarbage();
    auto &mirror = buffers[identity.key()];
    if (mirror.mapped and mirror.size == bytes.size() and mirror.mapped->mapped == bytes.data()) {
        return; // coherent persistent mapping: later commands see the CPU writes
    }
    if (mirror.mapped or mirror.buffer.get() == 0) {
        mirror = BufferMirror{};
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
    if (it->second.mapped and it->second.mapped->mapped == bytes.data()) {
        gl().glFinish(); // shader writes reach the mapping once the GPU is done
        return;
    }
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

GLuint OpenGLDevice::samplerFor(Filter const filter, Wrap const wrap) {
    auto const key = std::make_pair(filter, wrap);
    auto it = samplers.find(key);
    if (it == samplers.end()) {
        auto id = GLuint{0};
        gl().glGenSamplers(1, &id);
        auto sampler = GlObject{gl(), id, [](GlApi const &api, GLuint name) { api.glDeleteSamplers(1, &name); }};
        auto const glFilter = filter == Filter::Linear ? GL_LINEAR : GL_NEAREST;
        gl().glSamplerParameteri(id, GL_TEXTURE_MIN_FILTER, glFilter);
        gl().glSamplerParameteri(id, GL_TEXTURE_MAG_FILTER, glFilter);
        gl().glSamplerParameteri(id, GL_TEXTURE_WRAP_S, static_cast<GLint>(wrapMode(wrap)));
        gl().glSamplerParameteri(id, GL_TEXTURE_WRAP_T, static_cast<GLint>(wrapMode(wrap)));
        gl().glSamplerParameteri(id, GL_TEXTURE_WRAP_R, static_cast<GLint>(wrapMode(wrap)));
        it = samplers.emplace(key, std::move(sampler)).first;
    }
    return it->second.get();
}

void OpenGLDevice::bindTextureUnit(GLuint const unit, ImageMirror const &mirror, Filter const filter, Wrap const wrap) {
    gl().glActiveTexture(GL_TEXTURE0 + unit);
    gl().glBindTexture(textureTarget(mirror.dim), mirror.texture.get());
    gl().glBindSampler(unit, samplerFor(filter, wrap));
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
    auto const texelCount = static_cast<std::size_t>(transfer.extent.x) * transfer.extent.y * transfer.extent.z;
    auto const readbackSize = texelCount * 4 * sizeof(std::uint32_t);
    gl().glBindBuffer(GL_SHADER_STORAGE_BUFFER, readbackBuffer.get());
    if (readbackCapacity < readbackSize) {
        gl().glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(readbackSize), nullptr, GL_STREAM_READ);
        readbackCapacity = readbackSize;
    }
    gl().glMemoryBarrier(GL_ALL_BARRIER_BITS);
    gl().glUseProgram(readbackProgram(transfer.format, transfer.dim));
    bindTextureUnit(0, mirror, Filter::Nearest, Wrap::ClampToEdge);
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
    return programs.emplace(key, ProgramObject{*reflection, std::move(program), {}}).first->second;
}

GLuint OpenGLDevice::programForBoundExternals(ProgramObject &program) {
    auto formats = std::vector<std::uint32_t>{};
    for (auto const &slot : program.reflection.resources) {
        if (slot.kind != ResourceKind::ExternalTexture) {
            continue;
        }
        auto const bound = boundExternals.find(slot.binding);
        if (bound == boundExternals.end()) {
            throw DeviceError{"execute: external texture '" + std::string{slot.name} + "' (binding "
                              + std::to_string(slot.binding) + ") is not bound"};
        }
        formats.push_back(externals.at(bound->second).texture->format());
    }
    if (formats.empty()) {
        return program.program.get();
    }
    auto &variant = program.externalVariants[formats];
    if (variant.get() == 0) {
        auto const &reflection = program.reflection;
        variant = compileProgram(gl(), std::string{api == OpenGLApi::ES ? reflection.glslEs : reflection.glslDesktop},
                                 std::string{reflection.name});
    }
    return variant.get();
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
    boundTextures.clear();
    boundExternals.clear();
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

void OpenGLDevice::doBindTexture(TextureBindRequest const &request) {
    auto const it = images.find(request.identity->key());
    if (it == images.end() or not(it->second.format == request.format)) {
        throw DeviceError{"bindTexture: binding " + std::to_string(request.binding)
                          + " refers to a texture that was never uploaded"};
    }
    auto const isFloat32 = request.format.encoding == Encoding::Float and request.format.bitsPerChannel == 32;
    if (api == OpenGLApi::ES and request.filter == Filter::Linear and isFloat32
        and not hasGlExtension(gl(), "GL_OES_texture_float_linear")) {
        throw DeviceError{"bindTexture: linear filtering of 32-bit float textures needs GL_OES_texture_float_linear"};
    }
    boundTextures[request.binding] = BoundTexture{request.identity->key(), request.filter, request.wrap};
}

void OpenGLDevice::doBindExternalTexture(ExternalTextureBindRequest const &request) {
    egl->makeCurrent();
    collectGarbage();
    auto const &identity = request.image->identity();
    auto &mirror = externals[identity.key()];
    if (mirror.texture == nullptr) {
        mirror.owner = identity.watch();
        mirror.texture = importExternalTexture(*egl, *request.image);
    }
    boundExternals[request.binding] = identity.key();
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
    if (write.columns > 1) {
        // Default-block matrix uniforms take tightly packed columns.
        auto packed = std::vector<GLfloat>(write.columns * n);
        for (std::uint32_t c = 0; c < write.columns; ++c) {
            std::memcpy(packed.data() + c * n, data + c * write.columnStride, n * sizeof(GLfloat));
        }
        auto const *v = packed.data();
        switch (write.columns * 10 + n) {
        case 22: gl().glProgramUniformMatrix2fv(program, location, 1, GL_FALSE, v); break;
        case 33: gl().glProgramUniformMatrix3fv(program, location, 1, GL_FALSE, v); break;
        case 44: gl().glProgramUniformMatrix4fv(program, location, 1, GL_FALSE, v); break;
        case 23: gl().glProgramUniformMatrix2x3fv(program, location, 1, GL_FALSE, v); break;
        case 32: gl().glProgramUniformMatrix3x2fv(program, location, 1, GL_FALSE, v); break;
        case 24: gl().glProgramUniformMatrix2x4fv(program, location, 1, GL_FALSE, v); break;
        case 42: gl().glProgramUniformMatrix4x2fv(program, location, 1, GL_FALSE, v); break;
        case 34: gl().glProgramUniformMatrix3x4fv(program, location, 1, GL_FALSE, v); break;
        case 43: gl().glProgramUniformMatrix4x3fv(program, location, 1, GL_FALSE, v); break;
        default: throw DeviceError{"unsupported matrix uniform shape"};
        }
        return;
    }
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
    auto const program = programForBoundExternals(*current);
    gl().glUseProgram(program);
    for (auto const &slot : reflection.resources) {
        auto const name = std::string{slot.name} + "' (binding " + std::to_string(slot.binding) + ")";
        if (slot.kind == ResourceKind::StorageBuffer) {
            auto const bound = boundBuffers.find(slot.binding);
            if (bound == boundBuffers.end()) {
                throw DeviceError{"execute: buffer '" + name + " is not bound"};
            }
            gl().glBindBufferBase(GL_SHADER_STORAGE_BUFFER, slot.binding, buffers.at(bound->second).handle());
        } else if (slot.kind == ResourceKind::SampledTexture) {
            auto const bound = boundTextures.find(slot.binding);
            if (bound == boundTextures.end()) {
                throw DeviceError{"execute: texture '" + name + " is not bound"};
            }
            auto const &texture = bound->second;
            bindTextureUnit(slot.binding, images.at(texture.key), texture.filter, texture.wrap);
        } else if (slot.kind == ResourceKind::ExternalTexture) {
            gl().glActiveTexture(GL_TEXTURE0 + slot.binding);
            gl().glBindTexture(kTextureExternalOes, externals.at(boundExternals.at(slot.binding)).texture->texture());
            gl().glBindSampler(slot.binding, 0);
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
