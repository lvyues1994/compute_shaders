#include <tc/opengl/opengl_device.hpp>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <tc/detail/aliasing.hpp>
#include <tc/detail/hazards.hpp>

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

GlObject makeQuery(GlApi const &gl) {
    auto id = GLuint{0};
    gl.glGenQueries(1, &id);
    return GlObject{gl, id, [](GlApi const &api, GLuint name) { api.glDeleteQueries(1, &name); }};
}

// GL_TIME_ELAPSED queries; the entry points differ between desktop OpenGL
// and the OpenGL ES extension.
struct TimerQueries {
    PFNGLBEGINQUERYPROC begin = nullptr;
    PFNGLENDQUERYPROC end = nullptr;
    PfnGetQueryObjectUi64v result = nullptr;
    bool reportsDisjoint = false; // GL_GPU_DISJOINT_EXT
};

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
    return dim == Dim::D3 ? GL_TEXTURE_3D : dim == Dim::D2Array ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D;
}

// Internal formats of the compressed formats, and the extension (or GL
// version) that makes each family available: BC on desktop GL, ETC2 / EAC
// in OpenGL ES 3.0 and OpenGL 4.3, ASTC through GL_KHR_texture_compression_astc_ldr.
GLenum glCompressedFormat(CompressedFormat const format) {
    switch (format) {
    case CompressedFormat::Bc1: return 0x83F1;        // GL_COMPRESSED_RGBA_S3TC_DXT1_EXT
    case CompressedFormat::Bc3: return 0x83F3;        // GL_COMPRESSED_RGBA_S3TC_DXT5_EXT
    case CompressedFormat::Bc4: return 0x8DBB;        // GL_COMPRESSED_RED_RGTC1
    case CompressedFormat::Bc5: return 0x8DBD;        // GL_COMPRESSED_RG_RGTC2
    case CompressedFormat::Bc7: return 0x8E8C;        // GL_COMPRESSED_RGBA_BPTC_UNORM
    case CompressedFormat::Etc2Rgb8: return 0x9274;   // GL_COMPRESSED_RGB8_ETC2
    case CompressedFormat::Etc2Rgb8A1: return 0x9276; // GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2
    case CompressedFormat::Etc2Rgba8: return 0x9278;  // GL_COMPRESSED_RGBA8_ETC2_EAC
    case CompressedFormat::EacR11: return 0x9270;     // GL_COMPRESSED_R11_EAC
    case CompressedFormat::EacRg11: return 0x9272;    // GL_COMPRESSED_RG11_EAC
    case CompressedFormat::Astc4x4: return 0x93B0;    // GL_COMPRESSED_RGBA_ASTC_4x4_KHR
    case CompressedFormat::Astc5x5: return 0x93B2;
    case CompressedFormat::Astc6x6: return 0x93B4;
    case CompressedFormat::Astc8x8: return 0x93B7;
    case CompressedFormat::Astc10x10: return 0x93BB;
    case CompressedFormat::Astc12x12: return 0x93BD;
    }
    return GL_NONE;
}

// 3D images and arrays: allocated with glTexStorage3D, bound to image units
// with every layer.
bool isLayered(Dim const dim) {
    return dim == Dim::D3 or dim == Dim::D2Array;
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
    std::shared_ptr<GlObject> heap;       // set for placed device buffers: bytes [offset, offset + size) of it
    std::size_t offset = 0;
    std::size_t size = 0;

    GLuint handle() const noexcept { return mapped ? mapped->buffer : heap ? heap->get() : buffer.get(); }
};

// A memory heap: one buffer whose ranges are the placed device buffers.
struct HeapMirror {
    std::weak_ptr<void const> owner;
    std::shared_ptr<GlObject> buffer;
    std::size_t size = 0;
};

struct ImageMirror {
    std::weak_ptr<void const> owner;
    GlObject texture;
    ImageFormat format;
    Dim dim = Dim::D2;
    ivec3 extent{};
    std::uint32_t levels = 1;
    std::optional<CompressedFormat> compressed; // a block-compressed texture: `format` is unused
    std::map<std::uint32_t, GlObject> levelViews; // single-level views that image bindings write
};

struct BoundImage {
    void const *key = nullptr;
    std::uint32_t level = 0;
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
    Mip mip = Mip::None;
};

struct BoundBuffer {
    void const *key = nullptr;
    std::optional<ByteRange> range;
};

struct ExternalMirror {
    std::weak_ptr<void const> owner;
    std::unique_ptr<ExternalTexture> texture;
};

// An EGL fence after the work of one submit(). It keeps the EGL display
// alive, and any thread may wait on it without a current context.
struct EglFenceCompletion final : Submission::Completion {
    EglFenceCompletion(std::shared_ptr<EglContext> egl_, EGLSyncKHR const sync_) noexcept
        : egl(std::move(egl_)), sync(sync_) {}
    EglFenceCompletion(EglFenceCompletion const &) = delete;
    EglFenceCompletion &operator=(EglFenceCompletion const &) = delete;
    ~EglFenceCompletion() override { egl->sync().destroy(egl->eglDisplay(), sync); }

    bool isComplete() const override {
        if (signaled.load(std::memory_order_acquire)) {
            return true;
        }
        auto status = EGLint{0};
        if (egl->sync().getAttribute(egl->eglDisplay(), sync, EGL_SYNC_STATUS_KHR, &status) != EGL_TRUE) {
            throw DeviceError{"eglGetSyncAttribKHR failed"};
        }
        if (status != EGL_SIGNALED_KHR) {
            return false;
        }
        signaled.store(true, std::memory_order_release);
        return true;
    }

    void wait() const override {
        if (signaled.load(std::memory_order_acquire)) {
            return;
        }
        if (egl->sync().clientWait(egl->eglDisplay(), sync, 0, EGL_FOREVER_KHR) != EGL_CONDITION_SATISFIED_KHR) {
            throw DeviceError{"eglClientWaitSyncKHR failed"};
        }
        signaled.store(true, std::memory_order_release);
    }

    std::shared_ptr<EglContext> egl;
    EGLSyncKHR sync;
    mutable std::atomic<bool> signaled{false};
};

struct OpenGLDevice final : Device {
    explicit OpenGLDevice(OpenGLDeviceConfig const &config);

    DeviceInfo info() const override {
        auto result = DeviceInfo{};
        result.backend = api == OpenGLApi::ES ? "gles" : "opengl";
        result.deviceName = egl->renderer();
        result.timers = timerQueries.has_value();
        result.syncFiles = egl->sync().nativeFences and egl->sync().serverWaits;
        result.limits = limits;
        return result;
    }

protected:
    std::optional<ExternalMemory> doAllocateBuffer(ResourceIdentity const &identity, std::size_t bytes) override;
    void doUploadBuffer(ResourceIdentity const &identity, ConstByteSpan bytes, ByteRange range) override;
    void doDownloadBuffer(ResourceIdentity const &identity, ByteSpan bytes, ByteRange range) override;
    std::optional<std::string> doCheckKernel(KernelProgram const &program) const override {
        if (api == OpenGLApi::ES and program.glslEs.empty()) {
            return "kernel '" + std::string{program.name} + "' cannot run on OpenGL ES: "
                 + std::string{program.esUnavailableReason};
        }
        return std::nullopt;
    }
    void doUploadImage(ImageTransfer const &transfer) override;
    void doUploadCompressed(CompressedTransfer const &transfer) override;
    void doDownloadImage(ImageTransfer const &transfer) override;
    bool doCreateImage(ImageCreation const &request) override;
    void doCopyBufferToImage(BufferImageCopy const &copy) override;
    bool doCreateHeap(HeapCreation const &request) override;
    bool doCreateDeviceBuffer(BufferCreation const &request) override;
    PlacementSize doBufferPlacement(std::size_t const bytes) override {
        return PlacementSize{(bytes + 3) / 4 * 4, std::max<std::size_t>(limits.storageBufferOffsetAlignment, 16)};
    }
    // Images get memory of their own (DeviceInfo::placedImages is false);
    // the size only lets callers lay out heaps the same way everywhere.
    PlacementSize doImagePlacement(ImageCreation const &request) override {
        auto bytes = std::size_t{0};
        for (std::uint32_t level = 0; level < request.levels; ++level) {
            auto const extent = mipExtent(request.dim, request.extent, level);
            bytes += static_cast<std::size_t>(extent.x) * extent.y * extent.z * request.format.bytesPerTexel();
        }
        return PlacementSize{(bytes + 255) / 256 * 256, 256};
    }
    void doCopyBuffer(BufferCopy const &copy) override;
    void doUseKernel(KernelUse const &use) override;
    void doBindBuffer(BufferBindRequest const &request) override;
    void doBindImage(ImageBindRequest const &request) override;
    void doBindTexture(TextureBindRequest const &request) override;
    void doBindExternalTexture(ExternalTextureBindRequest const &request) override;
    void doBindExternalImage(ExternalImageBindRequest const &request) override;
    void doWaitFor(SyncFence fence) override;
    FencedSubmission doSubmitWithFence() override;
    void doSetUniform(UniformWrite const &write) override;
    void doExecute(Dispatch const &dispatch) override;
    Submission doSubmit() override;
    void doFinish() override {
        egl->makeCurrent();
        gl().glMemoryBarrier(GL_ALL_BARRIER_BITS);
        hazards.barrierRecorded();
        gl().glFinish();
    }
    std::vector<KernelTiming> doTakeTimings(std::uint64_t last) override;

private:
    struct PendingTiming {
        std::string program;
        uvec3 workGroups{};
        GlObject query;
        bool indirect = false;
        std::uint64_t submission = 0;
        bool disjoint = false; // a disjoint event was reported while it was pending
    };

    std::optional<TimerQueries> findTimerQueries() const;
    DeviceLimits queryLimits() const;
    bool gpuDisjoint() const;
    GlApi const &gl() const noexcept { return egl->api(); }
    PfnBufferStorage bufferStorage() const;
    PfnTextureView findTextureView() const;
    std::pair<GLuint, GLint> imageLevel(ImageMirror &mirror, std::uint32_t level);
    // Replaces `mirror` with new immutable storage.
    void allocateTexture(ImageMirror &mirror, ResourceIdentity const &identity, ImageFormat const &format,
                         std::optional<CompressedFormat> compressed, Dim dim, ivec3 extent, std::uint32_t levels);
    GLuint samplerFor(Filter filter, Wrap wrap, Mip mip);
    void bindTextureUnit(GLuint unit, ImageMirror const &mirror, Filter filter, Wrap wrap, Mip mip);
    void collectGarbage();
    ProgramObject &programFor(std::string_view name);
    GLuint programForBoundExternals(ProgramObject &program);
    GLuint readbackProgram(ImageFormat const &format, Dim dim);
    void applyUniform(GLuint program, UniformWrite const &write, std::vector<std::byte> const &bytes) const;
    void checkLimits(KernelProgram const &program, std::optional<uvec3> workGroups) const;

    std::shared_ptr<EglContext> egl; // declared first: destroyed after every GL object
    OpenGLApi api;
    std::unordered_map<void const *, HeapMirror> heaps;
    std::unordered_map<void const *, BufferMirror> buffers;
    std::unordered_map<void const *, ImageMirror> images;
    std::unordered_map<void const *, ExternalMirror> externals;
    std::unordered_map<void const *, ExternalMirror> externalImages; // frames bound as storage images
    std::unordered_map<std::string, ProgramObject> programs;
    std::map<std::string, GlObject> readbackPrograms;
    std::map<std::tuple<Filter, Wrap, Mip>, GlObject> samplers;
    GlObject readbackBuffer;
    std::size_t readbackCapacity = 0;

    ProgramObject *current = nullptr;
    std::map<std::uint32_t, BoundBuffer> boundBuffers;
    std::map<std::uint32_t, BoundImage> boundImages;
    std::map<std::uint32_t, BoundTexture> boundTextures;
    std::map<std::uint32_t, void const *> boundExternals;
    std::map<std::uint32_t, void const *> boundExternalImages;
    std::map<std::uint32_t, std::pair<UniformWrite, std::vector<std::byte>>> uniformValues;

    std::optional<TimerQueries> timerQueries;
    PfnTextureView textureView = nullptr;
    DeviceLimits limits;
    // Shader writes are incoherent: a glMemoryBarrier goes before a command
    // that uses what an earlier dispatch wrote (or writes what it read).
    // Placed buffers are tracked by their bytes of the heap.
    detail::HazardTracker hazards;
    detail::AliasTracker aliases;
    void barrierBefore(std::vector<detail::ResourceAccess> const &accesses) {
        if (hazards.needsBarrier(accesses)) {
            gl().glMemoryBarrier(GL_ALL_BARRIER_BITS);
        }
    }
    detail::ResourceAccess access(void const *const key, std::size_t const offset, std::size_t const size,
                                  bool const read, bool const write) const {
        return aliases.access(key, offset, size, read, write);
    }
    std::vector<GlObject> idleQueries;
    std::vector<PendingTiming> pendingTimings;
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
    timerQueries = findTimerQueries();
    textureView = findTextureView();
    limits = queryLimits();
    checkGl(gl(), "device setup");
}

DeviceLimits OpenGLDevice::queryLimits() const {
    auto const integer = [&](GLenum const name) {
        auto value = GLint{0};
        gl().glGetIntegerv(name, &value);
        return static_cast<std::uint32_t>(std::max(value, 0));
    };
    auto result = DeviceLimits{};
    result.storageBuffers = integer(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS);
    result.storageImages = integer(GL_MAX_COMPUTE_IMAGE_UNIFORMS);
    result.sampledTextures = integer(GL_MAX_COMPUTE_TEXTURE_IMAGE_UNITS);
    auto blockSize = GLint64{0};
    gl().glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &blockSize);
    result.storageBufferBytes = static_cast<std::uint64_t>(std::max<GLint64>(blockSize, 0));
    result.sharedBytes = integer(GL_MAX_COMPUTE_SHARED_MEMORY_SIZE);
    result.workGroupInvocations = integer(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS);
    for (GLuint axis = 0; axis < 3; ++axis) {
        auto size = GLint{0};
        auto count = GLint{0};
        gl().glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, axis, &size);
        gl().glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, axis, &count);
        result.workGroupSize[static_cast<int>(axis)] = static_cast<std::uint32_t>(size);
        result.workGroupCount[static_cast<int>(axis)] = static_cast<std::uint32_t>(count);
    }
    result.storageBufferOffsetAlignment = std::max(integer(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT), 1u);
    result.features = api == OpenGLApi::Desktop ? static_cast<std::uint32_t>(KernelFeature::Double) : 0u;
    // GL_KHR_shader_subgroup: every operation class tc exposes, in compute shaders.
    if (hasGlExtension(gl(), "GL_KHR_shader_subgroup")) {
        auto constexpr kSubgroupSize = GLenum{0x9532};
        auto constexpr kSubgroupStages = GLenum{0x9533};
        auto constexpr kSubgroupFeatures = GLenum{0x9534};
        auto constexpr needed = 0x3Fu; // basic, vote, arithmetic, ballot, shuffle, shuffle relative
        if ((integer(kSubgroupStages) & GL_COMPUTE_SHADER_BIT) != 0 and (integer(kSubgroupFeatures) & needed) == needed) {
            result.features |= static_cast<std::uint32_t>(KernelFeature::Subgroups);
            result.subgroupSize = integer(kSubgroupSize);
        }
    }
    auto const family = [&](CompressedFormat const first, CompressedFormat const last, bool const available) {
        for (auto i = static_cast<std::uint32_t>(first); available and i <= static_cast<std::uint32_t>(last); ++i) {
            result.compressedFormats |= compressedBit(static_cast<CompressedFormat>(i));
        }
    };
    auto const desktop = api == OpenGLApi::Desktop;
    auto const s3tc = hasGlExtension(gl(), "GL_EXT_texture_compression_s3tc");
    family(CompressedFormat::Bc1, CompressedFormat::Bc1, s3tc);
    family(CompressedFormat::Bc3, CompressedFormat::Bc3, s3tc);
    family(CompressedFormat::Bc4, CompressedFormat::Bc5, desktop or hasGlExtension(gl(), "GL_EXT_texture_compression_rgtc"));
    family(CompressedFormat::Bc7, CompressedFormat::Bc7, desktop or hasGlExtension(gl(), "GL_EXT_texture_compression_bptc"));
    family(CompressedFormat::Etc2Rgb8, CompressedFormat::EacRg11, true);
    family(CompressedFormat::Astc4x4, CompressedFormat::Astc12x12,
           hasGlExtension(gl(), "GL_KHR_texture_compression_astc_ldr"));
    // OpenGL ES cannot be asked which formats 2D arrays take; try each.
    for (std::uint32_t i = 0; i < kCompressedFormatCount; ++i) {
        auto const bit = compressedBit(static_cast<CompressedFormat>(i));
        if ((result.compressedFormats & bit) == 0) {
            continue;
        }
        while (gl().glGetError() != GL_NO_ERROR) {
        }
        auto const probe = makeTexture(gl());
        gl().glBindTexture(GL_TEXTURE_2D_ARRAY, probe.get());
        gl().glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, glCompressedFormat(static_cast<CompressedFormat>(i)), 4, 4, 1);
        if (gl().glGetError() == GL_NO_ERROR) {
            result.compressedArrayFormats |= bit;
        }
    }
    gl().glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
    if (api == OpenGLApi::Desktop and hasGlExtension(gl(), "GL_ARB_gpu_shader_int64")) {
        result.features |= static_cast<std::uint32_t>(KernelFeature::Int64);
        if (hasGlExtension(gl(), "GL_NV_shader_atomic_int64")) {
            result.features |= static_cast<std::uint32_t>(KernelFeature::Int64Atomics);
        }
    }
    result.imageDimension2D = integer(GL_MAX_TEXTURE_SIZE);
    result.imageDimension3D = integer(GL_MAX_3D_TEXTURE_SIZE);
    result.imageArrayLayers = integer(GL_MAX_ARRAY_TEXTURE_LAYERS);
    return result;
}

std::optional<TimerQueries> OpenGLDevice::findTimerQueries() const {
    auto queries = TimerQueries{};
    if (api == OpenGLApi::Desktop) {
        queries.begin = gl().glBeginQuery;
        queries.end = gl().glEndQuery;
        queries.result = gl().glGetQueryObjectui64v;
    } else if (hasGlExtension(gl(), "GL_EXT_disjoint_timer_query")) {
        queries.begin = gl().glBeginQueryEXT != nullptr ? gl().glBeginQueryEXT : gl().glBeginQuery;
        queries.end = gl().glEndQueryEXT != nullptr ? gl().glEndQueryEXT : gl().glEndQuery;
        queries.result = gl().glGetQueryObjectui64vEXT;
        queries.reportsDisjoint = true;
    }
    if (queries.result == nullptr) {
        return std::nullopt;
    }
    return queries;
}

// Reading GL_GPU_DISJOINT_EXT also clears it.
bool OpenGLDevice::gpuDisjoint() const {
    if (not timerQueries or not timerQueries->reportsDisjoint) {
        return false;
    }
    auto disjoint = GLint{0};
    gl().glGetIntegerv(kGpuDisjoint, &disjoint);
    return disjoint != 0;
}

// The queries of submissions up to `last` have results: their fences have
// signalled. Later queries stay pending.
std::vector<KernelTiming> OpenGLDevice::doTakeTimings(std::uint64_t const last) {
    egl->makeCurrent();
    // The flag does not say which queries the event affected.
    if (gpuDisjoint()) {
        for (auto &pending : pendingTimings) {
            pending.disjoint = true;
        }
    }
    auto result = std::vector<KernelTiming>{};
    auto kept = std::vector<PendingTiming>{};
    for (auto &pending : pendingTimings) {
        if (pending.submission > last) {
            kept.push_back(std::move(pending));
            continue;
        }
        auto nanoseconds = GLuint64{0};
        timerQueries->result(pending.query.get(), GL_QUERY_RESULT, &nanoseconds);
        auto timing = KernelTiming{std::move(pending.program), pending.workGroups, std::nullopt, pending.indirect,
                                   pending.submission};
        if (not pending.disjoint) {
            timing.milliseconds = static_cast<double>(nanoseconds) * 1e-6;
        }
        result.push_back(std::move(timing));
        idleQueries.push_back(std::move(pending.query));
    }
    pendingTimings = std::move(kept);
    checkGl(gl(), "takeTimings");
    return result;
}

// Drops device mirrors whose host resource no longer exists. A heap's
// buffer lives on while placed buffers use it.
void OpenGLDevice::collectGarbage() {
    for (auto it = heaps.begin(); it != heaps.end();) {
        it = it->second.owner.expired() ? heaps.erase(it) : std::next(it);
    }
    for (auto it = buffers.begin(); it != buffers.end();) {
        it = it->second.owner.expired() ? buffers.erase(it) : std::next(it);
    }
    for (auto it = images.begin(); it != images.end();) {
        it = it->second.owner.expired() ? images.erase(it) : std::next(it);
    }
    for (auto it = externals.begin(); it != externals.end();) {
        it = it->second.owner.expired() ? externals.erase(it) : std::next(it);
    }
    for (auto it = externalImages.begin(); it != externalImages.end();) {
        it = it->second.owner.expired() ? externalImages.erase(it) : std::next(it);
    }
}

PfnTextureView OpenGLDevice::findTextureView() const {
    if (api == OpenGLApi::Desktop) {
        return gl().glTextureView;
    }
    if (hasGlExtension(gl(), "GL_OES_texture_view")) {
        return gl().glTextureViewOES;
    }
    return hasGlExtension(gl(), "GL_EXT_texture_view") ? gl().glTextureViewEXT : nullptr;
}

// The texture and level that an image binding of `level` binds. Levels
// above 0 go through a view of that level where views exist: after a
// dispatch that samples a texture and writes one of its levels above 0,
// Adreno's GLES driver (830, V@0800) reads level n + 1 for every level n
// of that texture, in the dispatch and afterwards.
std::pair<GLuint, GLint> OpenGLDevice::imageLevel(ImageMirror &mirror, std::uint32_t const level) {
    if (level == 0 or textureView == nullptr) {
        return {mirror.texture.get(), static_cast<GLint>(level)};
    }
    auto it = mirror.levelViews.find(level);
    if (it == mirror.levelViews.end()) {
        auto view = makeTexture(gl());
        auto const layers = mirror.dim == Dim::D2Array ? static_cast<GLuint>(mirror.extent.z) : 1u;
        textureView(view.get(), textureTarget(mirror.dim), mirror.texture.get(), glFormatOf(mirror.format).internalFormat,
                    level, 1, 0, layers);
        checkGl(gl(), "glTextureView");
        it = mirror.levelViews.emplace(level, std::move(view)).first;
    }
    return {it->second.get(), 0};
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

void OpenGLDevice::doUploadBuffer(ResourceIdentity const &identity, ConstByteSpan const bytes, ByteRange const range) {
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
    barrierBefore({detail::ResourceAccess{identity.key(), range.offset, range.size, false, true}});
    gl().glBindBuffer(GL_SHADER_STORAGE_BUFFER, mirror.buffer.get());
    if (mirror.size != bytes.size()) {
        gl().glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(bytes.size()), bytes.data(),
                          GL_DYNAMIC_COPY);
        mirror.size = bytes.size();
    } else if (range.size > 0) {
        gl().glBufferSubData(GL_SHADER_STORAGE_BUFFER, static_cast<GLintptr>(range.offset),
                             static_cast<GLsizeiptr>(range.size), bytes.data() + range.offset);
    }
    checkGl(gl(), "uploadBuffer");
}

void OpenGLDevice::doDownloadBuffer(ResourceIdentity const &identity, ByteSpan const bytes, ByteRange const range) {
    egl->makeCurrent();
    auto const it = buffers.find(identity.key());
    if (it == buffers.end() or it->second.size != bytes.size()) {
        throw DeviceError{"downloadBuffer: the buffer was never uploaded to this device (or changed size)"};
    }
    gl().glMemoryBarrier(GL_ALL_BARRIER_BITS);
    hazards.barrierRecorded();
    if (it->second.mapped and it->second.mapped->mapped == bytes.data()) {
        gl().glFinish(); // shader writes reach the mapping once the GPU is done
        return;
    }
    if (range.size == 0) {
        return;
    }
    gl().glBindBuffer(GL_SHADER_STORAGE_BUFFER, it->second.buffer.get());
    auto const *mapped = gl().glMapBufferRange(GL_SHADER_STORAGE_BUFFER, static_cast<GLintptr>(range.offset),
                                               static_cast<GLsizeiptr>(range.size), GL_MAP_READ_BIT);
    if (mapped == nullptr) {
        checkGl(gl(), "downloadBuffer");
        throw DeviceError{"downloadBuffer: glMapBufferRange failed"};
    }
    std::memcpy(bytes.data() + range.offset, mapped, range.size);
    gl().glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    checkGl(gl(), "downloadBuffer");
}

void OpenGLDevice::allocateTexture(ImageMirror &mirror, ResourceIdentity const &identity, ImageFormat const &format,
                                   std::optional<CompressedFormat> const compressed, Dim const dim, ivec3 const extent,
                                   std::uint32_t const levels) {
    auto const target = textureTarget(dim);
    auto const internal = compressed ? glCompressedFormat(*compressed) : glFormatOf(format).internalFormat;
    mirror = ImageMirror{};
    mirror.owner = identity.watch();
    mirror.texture = makeTexture(gl());
    mirror.format = format;
    mirror.compressed = compressed;
    mirror.dim = dim;
    mirror.extent = extent;
    mirror.levels = levels;
    gl().glBindTexture(target, mirror.texture.get());
    if (isLayered(dim)) {
        gl().glTexStorage3D(target, static_cast<GLsizei>(levels), internal, extent.x, extent.y, extent.z);
    } else {
        gl().glTexStorage2D(target, static_cast<GLsizei>(levels), internal, extent.x, extent.y);
    }
    gl().glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    gl().glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

bool OpenGLDevice::doCreateImage(ImageCreation const &request) {
    egl->makeCurrent();
    collectGarbage();
    allocateTexture(images[request.identity->key()], *request.identity, request.format, request.compressed, request.dim,
                    request.extent, request.levels);
    checkGl(gl(), "createImage");
    return false;
}

bool OpenGLDevice::doCreateHeap(HeapCreation const &request) {
    egl->makeCurrent();
    collectGarbage();
    auto &heap = heaps[request.identity->key()];
    heap.owner = request.identity->watch();
    heap.buffer = std::make_shared<GlObject>(makeBuffer(gl()));
    heap.size = request.bytes;
    gl().glBindBuffer(GL_SHADER_STORAGE_BUFFER, heap.buffer->get());
    gl().glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(request.bytes), nullptr, GL_DYNAMIC_COPY);
    checkGl(gl(), "createHeap");
    return false;
}

bool OpenGLDevice::doCreateDeviceBuffer(BufferCreation const &request) {
    egl->makeCurrent();
    collectGarbage();
    auto mirror = BufferMirror{};
    mirror.owner = request.identity->watch();
    mirror.size = request.bytes;
    if (request.placement) {
        mirror.heap = heaps.at(request.placement->heap->key()).buffer;
        mirror.offset = request.placement->offset;
        aliases.place(request.identity->key(), request.identity->watch(),
                      detail::AliasTracker::Range{request.placement->heap->key(), mirror.offset, request.bytes});
    } else {
        mirror.buffer = makeBuffer(gl());
        gl().glBindBuffer(GL_SHADER_STORAGE_BUFFER, mirror.buffer.get());
        gl().glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(request.bytes), nullptr, GL_DYNAMIC_COPY);
    }
    buffers[request.identity->key()] = std::move(mirror);
    checkGl(gl(), "createDeviceBuffer");
    return false;
}

void OpenGLDevice::doCopyBuffer(BufferCopy const &copy) {
    egl->makeCurrent();
    auto const source = buffers.find(copy.source->key());
    auto const target = buffers.find(copy.target->key());
    if (source == buffers.end() or target == buffers.end()) {
        throw DeviceError{"copyBuffer: a buffer was never uploaded to this device"};
    }
    barrierBefore({access(copy.source->key(), copy.sourceOffset, copy.bytes, true, false),
                   access(copy.target->key(), copy.targetOffset, copy.bytes, false, true)});
    gl().glBindBuffer(GL_COPY_READ_BUFFER, source->second.handle());
    gl().glBindBuffer(GL_COPY_WRITE_BUFFER, target->second.handle());
    gl().glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER,
                             static_cast<GLintptr>(source->second.offset + copy.sourceOffset),
                             static_cast<GLintptr>(target->second.offset + copy.targetOffset),
                             static_cast<GLsizeiptr>(copy.bytes));
    checkGl(gl(), "copyBuffer");
}

// glTexSubImage from a pixel unpack buffer: the copy stays on the GPU.
void OpenGLDevice::doCopyBufferToImage(BufferImageCopy const &copy) {
    egl->makeCurrent();
    auto const buffer = buffers.find(copy.buffer->key());
    if (buffer == buffers.end() or buffer->second.size < copy.offset + copy.bytes) {
        throw DeviceError{"copyBufferToImage: the buffer was never uploaded to this device (or changed size)"};
    }
    auto &mirror = images.at(copy.image->key());
    barrierBefore({access(copy.buffer->key(), copy.offset, copy.bytes, true, false),
                   access(copy.image->key(), 0, 0, false, true)});
    auto const target = textureTarget(mirror.dim);
    auto const *offset =
        reinterpret_cast<void const *>(static_cast<std::uintptr_t>(buffer->second.offset + copy.offset));
    gl().glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer->second.handle());
    gl().glBindTexture(target, mirror.texture.get());
    gl().glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    auto const level = static_cast<GLint>(copy.level);
    if (copy.compressed) {
        auto const internal = glCompressedFormat(*copy.compressed);
        auto const bytes = static_cast<GLsizei>(copy.bytes);
        if (isLayered(mirror.dim)) {
            gl().glCompressedTexSubImage3D(target, level, 0, 0, static_cast<GLint>(copy.firstLayer), copy.extent.x,
                                           copy.extent.y, static_cast<GLsizei>(copy.layers), internal, bytes, offset);
        } else {
            gl().glCompressedTexSubImage2D(target, level, 0, 0, copy.extent.x, copy.extent.y, internal, bytes, offset);
        }
        gl().glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        checkGl(gl(), "copyBufferToImage");
        return;
    }
    auto const format = glFormatOf(mirror.format);
    if (isLayered(mirror.dim)) {
        gl().glTexSubImage3D(target, level, 0, 0, static_cast<GLint>(copy.firstLayer), copy.extent.x, copy.extent.y,
                             static_cast<GLsizei>(copy.layers), format.pixelFormat, format.pixelType, offset);
    } else {
        gl().glTexSubImage2D(target, level, 0, 0, copy.extent.x, copy.extent.y, format.pixelFormat, format.pixelType,
                             offset);
    }
    gl().glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    checkGl(gl(), "copyBufferToImage");
}

void OpenGLDevice::doUploadImage(ImageTransfer const &transfer) {
    egl->makeCurrent();
    collectGarbage();
    auto const format = glFormatOf(transfer.format);
    auto const target = textureTarget(transfer.dim);
    auto &mirror = images[transfer.identity->key()];
    auto const sameShape = mirror.texture.get() != 0 and mirror.format == transfer.format
                       and mirror.extent == transfer.extent and mirror.levels == transfer.levels;
    auto first = transfer.firstLevel;
    auto count = transfer.levelCount;
    if (not sameShape) {
        allocateTexture(mirror, *transfer.identity, transfer.format, std::nullopt, transfer.dim, transfer.extent,
                        transfer.levels);
        first = 0; // a new texture starts with every level
        count = transfer.levels;
    }
    barrierBefore({detail::ResourceAccess{transfer.identity->key(), 0, 0, false, true}});
    gl().glBindTexture(target, mirror.texture.get());
    gl().glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    for (auto level = first; level < first + count; ++level) {
        auto const extent = mipExtent(transfer.dim, transfer.extent, level);
        auto const texelCount = static_cast<std::size_t>(extent.x) * extent.y * extent.z;
        auto texels = std::vector<std::byte>(texelCount * transfer.format.bytesPerTexel());
        transfer.writeTexels(level, ByteSpan{texels.data(), texels.size()});
        auto const glLevel = static_cast<GLint>(level);
        if (isLayered(transfer.dim)) {
            gl().glTexSubImage3D(target, glLevel, 0, 0, 0, extent.x, extent.y, extent.z, format.pixelFormat,
                                 format.pixelType, texels.data());
        } else {
            gl().glTexSubImage2D(target, glLevel, 0, 0, extent.x, extent.y, format.pixelFormat, format.pixelType,
                                 texels.data());
        }
    }
    checkGl(gl(), "uploadImage");
}

void OpenGLDevice::doUploadCompressed(CompressedTransfer const &transfer) {
    egl->makeCurrent();
    collectGarbage();
    if ((limits.compressedFormats & compressedBit(transfer.format)) == 0) {
        throw DeviceError{std::string{"uploadImage: "} + egl->renderer() + " does not sample "
                          + compressedFormatInfo(transfer.format).name + " textures"};
    }
    auto const internal = glCompressedFormat(transfer.format);
    auto &mirror = images[transfer.identity->key()];
    auto const extent3 = ivec3{transfer.extent, static_cast<std::int32_t>(transfer.layers)};
    if (mirror.texture.get() == 0 or mirror.compressed != transfer.format or mirror.dim != transfer.dim
        or not(mirror.extent == extent3) or mirror.levels != transfer.levels) {
        allocateTexture(mirror, *transfer.identity, ImageFormat{}, transfer.format, transfer.dim, extent3,
                        transfer.levels);
    }
    barrierBefore({detail::ResourceAccess{transfer.identity->key(), 0, 0, false, true}});
    auto const target = textureTarget(transfer.dim);
    gl().glBindTexture(target, mirror.texture.get());
    for (std::uint32_t level = 0; level < transfer.levels; ++level) {
        auto const size = transfer.image->extent(level);
        auto const &blocks = transfer.image->blocks(level);
        auto const bytes = static_cast<GLsizei>(blocks.size());
        if (transfer.dim == Dim::D2Array) {
            gl().glCompressedTexSubImage3D(target, static_cast<GLint>(level), 0, 0, 0, size.x, size.y,
                                           static_cast<GLsizei>(transfer.layers), internal, bytes, blocks.data());
        } else {
            gl().glCompressedTexSubImage2D(target, static_cast<GLint>(level), 0, 0, size.x, size.y, internal, bytes,
                                           blocks.data());
        }
    }
    checkGl(gl(), "uploadImage (compressed)");
}

GLuint OpenGLDevice::samplerFor(Filter const filter, Wrap const wrap, Mip const mip) {
    auto const key = std::make_tuple(filter, wrap, mip);
    auto it = samplers.find(key);
    if (it == samplers.end()) {
        auto id = GLuint{0};
        gl().glGenSamplers(1, &id);
        auto sampler = GlObject{gl(), id, [](GlApi const &api, GLuint name) { api.glDeleteSamplers(1, &name); }};
        auto const linear = filter == Filter::Linear;
        auto const minFilter = mip == Mip::None      ? (linear ? GL_LINEAR : GL_NEAREST)
                             : mip == Mip::Nearest ? (linear ? GL_LINEAR_MIPMAP_NEAREST : GL_NEAREST_MIPMAP_NEAREST)
                                                   : (linear ? GL_LINEAR_MIPMAP_LINEAR : GL_NEAREST_MIPMAP_LINEAR);
        gl().glSamplerParameteri(id, GL_TEXTURE_MIN_FILTER, minFilter);
        gl().glSamplerParameteri(id, GL_TEXTURE_MAG_FILTER, linear ? GL_LINEAR : GL_NEAREST);
        gl().glSamplerParameteri(id, GL_TEXTURE_WRAP_S, static_cast<GLint>(wrapMode(wrap)));
        gl().glSamplerParameteri(id, GL_TEXTURE_WRAP_T, static_cast<GLint>(wrapMode(wrap)));
        gl().glSamplerParameteri(id, GL_TEXTURE_WRAP_R, static_cast<GLint>(wrapMode(wrap)));
        it = samplers.emplace(key, std::move(sampler)).first;
    }
    return it->second.get();
}

void OpenGLDevice::bindTextureUnit(GLuint const unit, ImageMirror const &mirror, Filter const filter, Wrap const wrap,
                                   Mip const mip) {
    gl().glActiveTexture(GL_TEXTURE0 + unit);
    gl().glBindTexture(textureTarget(mirror.dim), mirror.texture.get());
    gl().glBindSampler(unit, samplerFor(filter, wrap, mip));
}

GLuint OpenGLDevice::readbackProgram(ImageFormat const &format, Dim const dim) {
    auto const key = glslFormatQualifier(format) + "_" + std::to_string(static_cast<int>(dim));
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
    if (it == images.end() or not(it->second.format == transfer.format) or not(it->second.extent == transfer.extent)
        or it->second.levels != transfer.levels) {
        throw DeviceError{"downloadImage: the image was never uploaded to this device"};
    }
    auto const &mirror = it->second;
    auto const program = readbackProgram(transfer.format, transfer.dim);
    gl().glMemoryBarrier(GL_ALL_BARRIER_BITS);
    hazards.barrierRecorded();
    gl().glUseProgram(program);
    bindTextureUnit(0, mirror, Filter::Nearest, Wrap::ClampToEdge, Mip::Nearest);
    for (auto level = transfer.firstLevel; level < transfer.firstLevel + transfer.levelCount; ++level) {
        auto const extent = mipExtent(transfer.dim, transfer.extent, level);
        auto const texelCount = static_cast<std::size_t>(extent.x) * extent.y * extent.z;
        auto const readbackSize = texelCount * 4 * sizeof(std::uint32_t);
        gl().glBindBuffer(GL_SHADER_STORAGE_BUFFER, readbackBuffer.get());
        if (readbackCapacity < readbackSize) {
            gl().glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(readbackSize), nullptr,
                              GL_STREAM_READ);
            readbackCapacity = readbackSize;
        }
        auto const glLevel = static_cast<GLint>(level);
        gl().glProgramUniform1iv(program, 0, 1, &glLevel);
        gl().glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, readbackBuffer.get());
        gl().glDispatchCompute((static_cast<GLuint>(extent.x) + 7u) / 8u, (static_cast<GLuint>(extent.y) + 7u) / 8u,
                               static_cast<GLuint>(extent.z));
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
            std::uint32_t const components[4] = {mapped[4 * i], mapped[4 * i + 1], mapped[4 * i + 2],
                                                 mapped[4 * i + 3]};
            encodeTexel(transfer.format, components, texels.data() + i * texelBytes);
        }
        gl().glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        checkGl(gl(), "downloadImage");
        transfer.readTexels(level, ConstByteSpan{texels.data(), texels.size()});
    }
}

ProgramObject &OpenGLDevice::programFor(std::string_view const name) {
    auto const key = std::string{name};
    auto const reflection = ProgramRegistry::instance().find(name);
    if (auto const it = programs.find(key); it != programs.end()) {
        if (not reflection or it->second.reflection.revision == reflection->revision) {
            return it->second;
        }
        programs.erase(it); // reloaded from disk: compile the new revision
    }
    if (not reflection) {
        throw DeviceError{"kernel '" + key + "' was not transpiled; add its header with tc_add_kernels()"};
    }
    auto const source = api == OpenGLApi::ES ? reflection->glslEs : reflection->glslDesktop;
    if (source.empty()) {
        throw DeviceError{"kernel '" + key + "' cannot run on OpenGL ES: " + std::string{reflection->esUnavailableReason}};
    }
    if (auto const reason = exceedsLimits(*reflection, limits)) {
        throw DeviceError{*reason};
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
    boundExternalImages.clear();
    uniformValues.clear();
}

void OpenGLDevice::doBindBuffer(BufferBindRequest const &request) {
    auto const it = buffers.find(request.identity->key());
    if (it == buffers.end()) {
        throw DeviceError{"bindBuffer: binding " + std::to_string(request.binding)
                          + " refers to a buffer that was never uploaded"};
    }
    if (request.range) {
        if (request.range->offset % limits.storageBufferOffsetAlignment != 0) {
            throw DeviceError{"bindBuffer: binding " + std::to_string(request.binding) + " starts at byte "
                              + std::to_string(request.range->offset) + ", not a multiple of "
                              + std::to_string(limits.storageBufferOffsetAlignment)
                              + " (DeviceLimits::storageBufferOffsetAlignment)"};
        }
        if (request.range->offset + request.range->size > it->second.size) {
            throw DeviceError{"bindBuffer: binding " + std::to_string(request.binding)
                              + " extends past the device buffer"};
        }
    }
    boundBuffers[request.binding] = BoundBuffer{request.identity->key(), request.range};
}

void OpenGLDevice::doBindImage(ImageBindRequest const &request) {
    auto const it = images.find(request.identity->key());
    if (it == images.end() or not(it->second.format == request.format) or request.level >= it->second.levels) {
        throw DeviceError{"bindImage: binding " + std::to_string(request.binding)
                          + " refers to an image that was never uploaded"};
    }
    boundImages[request.binding] = BoundImage{request.identity->key(), request.level};
}

void OpenGLDevice::doBindTexture(TextureBindRequest const &request) {
    auto const it = images.find(request.identity->key());
    auto const matches = it != images.end() and it->second.dim == request.dim
                     and (request.compressed ? it->second.compressed == request.compressed
                                             : not it->second.compressed and it->second.format == request.format);
    if (not matches) {
        throw DeviceError{"bindTexture: binding " + std::to_string(request.binding)
                          + " refers to a texture that was never uploaded"};
    }
    auto const isFloat32 = request.format.encoding == Encoding::Float and request.format.bitsPerChannel == 32;
    auto const filtered = request.filter == Filter::Linear or request.mip == Mip::Linear;
    if (api == OpenGLApi::ES and filtered and isFloat32 and not hasGlExtension(gl(), "GL_OES_texture_float_linear")) {
        throw DeviceError{"bindTexture: linear filtering of 32-bit float textures needs GL_OES_texture_float_linear; "
                          "16-bit float textures filter everywhere"};
    }
    boundTextures[request.binding] =
        BoundTexture{request.identity->key(), request.filter, request.wrap, request.mip};
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

void OpenGLDevice::doBindExternalImage(ExternalImageBindRequest const &request) {
    egl->makeCurrent();
    collectGarbage();
    auto const &identity = request.image->identity();
    auto &mirror = externalImages[identity.key()];
    if (mirror.texture == nullptr) {
        mirror.owner = identity.watch();
        mirror.texture = importExternalStorage(*egl, *request.image);
    }
    boundExternalImages[request.binding] = identity.key();
}

// A GPU-side wait where EGL can import the sync file, else a host wait.
void OpenGLDevice::doWaitFor(SyncFence fence) {
    if (fence.empty()) {
        return;
    }
    auto const &sync = egl->sync();
    if (not sync.nativeFences or not sync.serverWaits) {
        fence.wait();
        return;
    }
    egl->makeCurrent();
    EGLint const attributes[] = {EGL_SYNC_NATIVE_FENCE_FD_ANDROID, fence.get(), EGL_NONE};
    auto const imported = sync.create(egl->eglDisplay(), EGL_SYNC_NATIVE_FENCE_ANDROID, attributes);
    if (imported == EGL_NO_SYNC_KHR) {
        fence.wait();
        return;
    }
    fence.release(); // EGL owns the descriptor now
    sync.serverWait(egl->eglDisplay(), imported, 0);
    sync.destroy(egl->eglDisplay(), imported);
}

FencedSubmission OpenGLDevice::doSubmitWithFence() {
    auto const &sync = egl->sync();
    if (not sync.nativeFences) {
        auto submission = doSubmit();
        submission.wait();
        return FencedSubmission{std::move(submission), SyncFence{}};
    }
    egl->makeCurrent();
    gl().glMemoryBarrier(GL_ALL_BARRIER_BITS);
    hazards.barrierRecorded();
    EGLint const attributes[] = {EGL_SYNC_NATIVE_FENCE_FD_ANDROID, EGL_NO_NATIVE_FENCE_FD_ANDROID, EGL_NONE};
    auto const fence = sync.create(egl->eglDisplay(), EGL_SYNC_NATIVE_FENCE_ANDROID, attributes);
    if (fence == EGL_NO_SYNC_KHR) {
        throw DeviceError{"eglCreateSyncKHR (native fence) failed"};
    }
    gl().glFlush(); // the descriptor exists once the fence is flushed
    auto const fd = sync.dupNativeFence(egl->eglDisplay(), fence);
    // The host waits on the same sync object.
    return FencedSubmission{Submission{std::make_shared<EglFenceCompletion>(egl, fence)},
                            SyncFence{fd == EGL_NO_NATIVE_FENCE_FD_ANDROID ? -1 : fd}};
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

void OpenGLDevice::checkLimits(KernelProgram const &program, std::optional<uvec3> const workGroups) const {
    auto invocations = GLint{0};
    gl().glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &invocations);
    auto const requested = program.localSize.x * program.localSize.y * program.localSize.z;
    if (requested > static_cast<uint>(invocations)) {
        throw DeviceError{"kernel '" + std::string{program.name} + "' uses " + std::to_string(requested)
                          + " invocations per workgroup; this device allows " + std::to_string(invocations)};
    }
    for (GLuint axis = 0; axis < 3 and workGroups; ++axis) {
        auto count = GLint{0};
        gl().glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, axis, &count);
        if ((*workGroups)[static_cast<int>(axis)] > static_cast<uint>(count)) {
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
    checkLimits(reflection, dispatch.indirect ? std::nullopt : std::optional<uvec3>{dispatch.workGroups});
    auto indirectOffset = std::size_t{0};
    if (dispatch.indirect) {
        auto const it = buffers.find(dispatch.indirect->identity->key());
        if (it == buffers.end() or it->second.size < dispatch.indirect->offset + sizeof(DispatchIndirectCommand)) {
            throw DeviceError{"executeIndirect: the argument buffer was never uploaded to this device"};
        }
        gl().glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, it->second.handle());
        indirectOffset = it->second.offset + dispatch.indirect->offset;
    }
    auto const program = programForBoundExternals(*current);
    gl().glUseProgram(program);
    auto accesses = std::vector<detail::ResourceAccess>{};
    auto const use = [&](void const *const key, std::optional<ByteRange> const &range, Access const how) {
        accesses.push_back(access(key, range ? range->offset : 0, range ? range->size : 0, how != Access::WriteOnly,
                                  how != Access::ReadOnly));
    };
    if (dispatch.indirect) {
        use(dispatch.indirect->identity->key(), ByteRange{dispatch.indirect->offset, sizeof(DispatchIndirectCommand)},
            Access::ReadOnly);
    }
    for (auto const &slot : reflection.resources) {
        auto const name = std::string{slot.name} + "' (binding " + std::to_string(slot.binding) + ")";
        if (slot.kind == ResourceKind::StorageBuffer) {
            auto const bound = boundBuffers.find(slot.binding);
            if (bound == boundBuffers.end()) {
                throw DeviceError{"execute: buffer '" + name + " is not bound"};
            }
            use(bound->second.key, bound->second.range, slot.access);
            auto const &mirror = buffers.at(bound->second.key);
            auto const &range = bound->second.range;
            if (range or mirror.heap) {
                auto const offset = mirror.offset + (range ? range->offset : 0);
                auto const size = range ? range->size : mirror.size;
                gl().glBindBufferRange(GL_SHADER_STORAGE_BUFFER, slot.binding, mirror.handle(),
                                       static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(size));
            } else {
                gl().glBindBufferBase(GL_SHADER_STORAGE_BUFFER, slot.binding, mirror.handle());
            }
        } else if (slot.kind == ResourceKind::SampledTexture) {
            auto const bound = boundTextures.find(slot.binding);
            if (bound == boundTextures.end()) {
                throw DeviceError{"execute: texture '" + name + " is not bound"};
            }
            auto const &texture = bound->second;
            use(texture.key, std::nullopt, Access::ReadOnly);
            bindTextureUnit(slot.binding, images.at(texture.key), texture.filter, texture.wrap, texture.mip);
        } else if (slot.kind == ResourceKind::ExternalImage) {
            auto const bound = boundExternalImages.find(slot.binding);
            if (bound == boundExternalImages.end()) {
                throw DeviceError{"execute: external image '" + name + " is not bound"};
            }
            use(bound->second, std::nullopt, slot.access);
            gl().glBindImageTexture(slot.binding, externalImages.at(bound->second).texture->texture(), 0, GL_FALSE, 0,
                                    imageAccess(slot.access), glFormatOf(slot.imageFormat).internalFormat);
        } else if (slot.kind == ResourceKind::ExternalTexture) {
            gl().glActiveTexture(GL_TEXTURE0 + slot.binding);
            gl().glBindTexture(kTextureExternalOes, externals.at(boundExternals.at(slot.binding)).texture->texture());
            gl().glBindSampler(slot.binding, 0);
        } else {
            auto const bound = boundImages.find(slot.binding);
            if (bound == boundImages.end()) {
                throw DeviceError{"execute: image '" + name + " is not bound"};
            }
            use(bound->second.key, std::nullopt, slot.access);
            auto &mirror = images.at(bound->second.key);
            auto const [texture, level] = imageLevel(mirror, bound->second.level);
            gl().glBindImageTexture(slot.binding, texture, level, isLayered(mirror.dim) ? GL_TRUE : GL_FALSE, 0,
                                    imageAccess(slot.access), glFormatOf(mirror.format).internalFormat);
        }
    }
    for (auto const &slot : reflection.uniforms) {
        auto const value = uniformValues.find(slot.location);
        if (value == uniformValues.end()) {
            throw DeviceError{"execute: uniform '" + std::string{slot.name} + "' was never set with setUniform()"};
        }
        applyUniform(program, value->second.first, value->second.second);
    }
    auto const record = [&] {
        if (dispatch.indirect) {
            gl().glDispatchComputeIndirect(static_cast<GLintptr>(indirectOffset));
        } else {
            gl().glDispatchCompute(dispatch.workGroups.x, dispatch.workGroups.y, dispatch.workGroups.z);
        }
    };
    if (timersEnabled()) {
        // Timed dispatches run one at a time.
        gl().glMemoryBarrier(GL_ALL_BARRIER_BITS);
        hazards.barrierRecorded();
    }
    barrierBefore(accesses);
    if (not timersEnabled()) {
        record();
        checkGl(gl(), "execute");
        return;
    }
    if (pendingTimings.empty()) {
        gpuDisjoint(); // clears a disjoint event from before this measurement
    }
    auto pending = PendingTiming{std::string{dispatch.programName}, dispatch.indirect ? uvec3{} : dispatch.workGroups,
                                 GlObject{}, dispatch.indirect.has_value(), nextSubmission()};
    if (idleQueries.empty()) {
        pending.query = makeQuery(gl());
    } else {
        pending.query = std::move(idleQueries.back());
        idleQueries.pop_back();
    }
    timerQueries->begin(kTimeElapsed, pending.query.get());
    record();
    timerQueries->end(kTimeElapsed);
    pendingTimings.push_back(std::move(pending));
    checkGl(gl(), "execute");
}

// Shader writes become visible to mapped memory, then an EGL fence marks
// the end of the work so far.
Submission OpenGLDevice::doSubmit() {
    egl->makeCurrent();
    gl().glMemoryBarrier(GL_ALL_BARRIER_BITS);
    hazards.barrierRecorded();
    auto const &sync = egl->sync();
    if (not sync.fences) {
        gl().glFinish();
        return Submission{};
    }
    auto const fence = sync.create(egl->eglDisplay(), EGL_SYNC_FENCE_KHR, nullptr);
    if (fence == EGL_NO_SYNC_KHR) {
        throw DeviceError{"eglCreateSyncKHR failed"};
    }
    gl().glFlush();
    return Submission{std::make_shared<EglFenceCompletion>(egl, fence)};
}

} // namespace

std::unique_ptr<Device> createDevice(OpenGLDeviceConfig const &config) {
    return std::make_unique<OpenGLDevice>(config);
}

} // namespace tc::opengl
