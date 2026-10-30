#pragma once

#include <GLES3/gl32.h>

namespace tc::opengl {

// glProgramUniform*dv exist only in desktop OpenGL.
using PfnProgramUniformDv = void(GL_APIENTRY *)(GLuint, GLint, GLsizei, double const *);
// glBufferStorage: OpenGL 4.4, or GL_EXT_buffer_storage on OpenGL ES.
using PfnBufferStorage = void(GL_APIENTRY *)(GLenum, GLsizeiptr, void const *, GLbitfield);

inline constexpr GLbitfield kMapPersistentBit = 0x0040;
inline constexpr GLbitfield kMapCoherentBit = 0x0080;

// Timer queries: core in desktop OpenGL 3.3, GL_EXT_disjoint_timer_query on
// OpenGL ES.
using PfnGetQueryObjectUi64v = void(GL_APIENTRY *)(GLuint, GLenum, GLuint64 *);
inline constexpr GLenum kTimeElapsed = 0x88BF;
inline constexpr GLenum kGpuDisjoint = 0x8FBB;

// Entry points of the OpenGL 4.3 core / OpenGL ES 3.1 common subset that the
// backend uses, loaded at runtime so one binary drives both APIs.
#define TC_GL_REQUIRED_FUNCTIONS(X)                                                                                    \
    X(PFNGLGETSTRINGPROC, glGetString)                                                                                 \
    X(PFNGLGETERRORPROC, glGetError)                                                                                   \
    X(PFNGLGETINTEGERVPROC, glGetIntegerv)                                                                             \
    X(PFNGLGETINTEGERI_VPROC, glGetIntegeri_v)                                                                         \
    X(PFNGLCREATESHADERPROC, glCreateShader)                                                                           \
    X(PFNGLSHADERSOURCEPROC, glShaderSource)                                                                           \
    X(PFNGLCOMPILESHADERPROC, glCompileShader)                                                                         \
    X(PFNGLGETSHADERIVPROC, glGetShaderiv)                                                                             \
    X(PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog)                                                                   \
    X(PFNGLDELETESHADERPROC, glDeleteShader)                                                                           \
    X(PFNGLCREATEPROGRAMPROC, glCreateProgram)                                                                         \
    X(PFNGLATTACHSHADERPROC, glAttachShader)                                                                           \
    X(PFNGLLINKPROGRAMPROC, glLinkProgram)                                                                             \
    X(PFNGLGETPROGRAMIVPROC, glGetProgramiv)                                                                           \
    X(PFNGLGETPROGRAMINFOLOGPROC, glGetProgramInfoLog)                                                                 \
    X(PFNGLDELETEPROGRAMPROC, glDeleteProgram)                                                                         \
    X(PFNGLUSEPROGRAMPROC, glUseProgram)                                                                               \
    X(PFNGLGENBUFFERSPROC, glGenBuffers)                                                                               \
    X(PFNGLDELETEBUFFERSPROC, glDeleteBuffers)                                                                         \
    X(PFNGLBINDBUFFERPROC, glBindBuffer)                                                                               \
    X(PFNGLBUFFERDATAPROC, glBufferData)                                                                               \
    X(PFNGLBUFFERSUBDATAPROC, glBufferSubData)                                                                         \
    X(PFNGLMAPBUFFERRANGEPROC, glMapBufferRange)                                                                       \
    X(PFNGLUNMAPBUFFERPROC, glUnmapBuffer)                                                                             \
    X(PFNGLBINDBUFFERBASEPROC, glBindBufferBase)                                                                       \
    X(PFNGLGENTEXTURESPROC, glGenTextures)                                                                             \
    X(PFNGLDELETETEXTURESPROC, glDeleteTextures)                                                                       \
    X(PFNGLBINDTEXTUREPROC, glBindTexture)                                                                             \
    X(PFNGLTEXPARAMETERIPROC, glTexParameteri)                                                                         \
    X(PFNGLTEXSTORAGE2DPROC, glTexStorage2D)                                                                           \
    X(PFNGLTEXSTORAGE3DPROC, glTexStorage3D)                                                                           \
    X(PFNGLTEXSUBIMAGE2DPROC, glTexSubImage2D)                                                                         \
    X(PFNGLTEXSUBIMAGE3DPROC, glTexSubImage3D)                                                                         \
    X(PFNGLPIXELSTOREIPROC, glPixelStorei)                                                                             \
    X(PFNGLBINDIMAGETEXTUREPROC, glBindImageTexture)                                                                   \
    X(PFNGLDISPATCHCOMPUTEPROC, glDispatchCompute)                                                                     \
    X(PFNGLMEMORYBARRIERPROC, glMemoryBarrier)                                                                         \
    X(PFNGLFINISHPROC, glFinish)                                                                                       \
    X(PFNGLPROGRAMUNIFORM1IVPROC, glProgramUniform1iv)                                                                 \
    X(PFNGLPROGRAMUNIFORM2IVPROC, glProgramUniform2iv)                                                                 \
    X(PFNGLPROGRAMUNIFORM3IVPROC, glProgramUniform3iv)                                                                 \
    X(PFNGLPROGRAMUNIFORM4IVPROC, glProgramUniform4iv)                                                                 \
    X(PFNGLPROGRAMUNIFORM1UIVPROC, glProgramUniform1uiv)                                                               \
    X(PFNGLPROGRAMUNIFORM2UIVPROC, glProgramUniform2uiv)                                                               \
    X(PFNGLPROGRAMUNIFORM3UIVPROC, glProgramUniform3uiv)                                                               \
    X(PFNGLPROGRAMUNIFORM4UIVPROC, glProgramUniform4uiv)                                                               \
    X(PFNGLPROGRAMUNIFORM1FVPROC, glProgramUniform1fv)                                                                 \
    X(PFNGLPROGRAMUNIFORM2FVPROC, glProgramUniform2fv)                                                                 \
    X(PFNGLPROGRAMUNIFORM3FVPROC, glProgramUniform3fv)                                                                 \
    X(PFNGLPROGRAMUNIFORM4FVPROC, glProgramUniform4fv)                                                                 \
    X(PFNGLPROGRAMUNIFORMMATRIX2FVPROC, glProgramUniformMatrix2fv)                                                     \
    X(PFNGLPROGRAMUNIFORMMATRIX3FVPROC, glProgramUniformMatrix3fv)                                                     \
    X(PFNGLPROGRAMUNIFORMMATRIX4FVPROC, glProgramUniformMatrix4fv)                                                     \
    X(PFNGLPROGRAMUNIFORMMATRIX2X3FVPROC, glProgramUniformMatrix2x3fv)                                                 \
    X(PFNGLPROGRAMUNIFORMMATRIX3X2FVPROC, glProgramUniformMatrix3x2fv)                                                 \
    X(PFNGLPROGRAMUNIFORMMATRIX2X4FVPROC, glProgramUniformMatrix2x4fv)                                                 \
    X(PFNGLPROGRAMUNIFORMMATRIX4X2FVPROC, glProgramUniformMatrix4x2fv)                                                 \
    X(PFNGLPROGRAMUNIFORMMATRIX3X4FVPROC, glProgramUniformMatrix3x4fv)                                                 \
    X(PFNGLPROGRAMUNIFORMMATRIX4X3FVPROC, glProgramUniformMatrix4x3fv)                                                 \
    X(PFNGLACTIVETEXTUREPROC, glActiveTexture)                                                                         \
    X(PFNGLGENSAMPLERSPROC, glGenSamplers)                                                                             \
    X(PFNGLDELETESAMPLERSPROC, glDeleteSamplers)                                                                       \
    X(PFNGLSAMPLERPARAMETERIPROC, glSamplerParameteri)                                                                 \
    X(PFNGLBINDSAMPLERPROC, glBindSampler)                                                                             \
    X(PFNGLGETSTRINGIPROC, glGetStringi)                                                                               \
    X(PFNGLGENQUERIESPROC, glGenQueries)                                                                               \
    X(PFNGLDELETEQUERIESPROC, glDeleteQueries)                                                                         \
    X(PFNGLBEGINQUERYPROC, glBeginQuery)                                                                               \
    X(PFNGLENDQUERYPROC, glEndQuery)

#define TC_GL_OPTIONAL_FUNCTIONS(X)                                                                                    \
    X(PFNGLDEBUGMESSAGECALLBACKPROC, glDebugMessageCallback)                                                           \
    X(PFNGLENABLEPROC, glEnable)                                                                                       \
    X(PfnProgramUniformDv, glProgramUniform1dv)                                                                        \
    X(PfnProgramUniformDv, glProgramUniform2dv)                                                                        \
    X(PfnProgramUniformDv, glProgramUniform3dv)                                                                        \
    X(PfnProgramUniformDv, glProgramUniform4dv)                                                                        \
    X(PfnBufferStorage, glBufferStorage)                                                                               \
    X(PfnBufferStorage, glBufferStorageEXT)                                                                            \
    X(PfnGetQueryObjectUi64v, glGetQueryObjectui64v)                                                                   \
    X(PfnGetQueryObjectUi64v, glGetQueryObjectui64vEXT)                                                                \
    X(PFNGLBEGINQUERYPROC, glBeginQueryEXT)                                                                            \
    X(PFNGLENDQUERYPROC, glEndQueryEXT)

struct GlApi {
#define TC_GL_DECLARE(type, name) type name = nullptr;
    TC_GL_REQUIRED_FUNCTIONS(TC_GL_DECLARE)
    TC_GL_OPTIONAL_FUNCTIONS(TC_GL_DECLARE)
#undef TC_GL_DECLARE

    // Loads every entry point of the current context; throws DeviceError
    // naming the first missing required function.
    static GlApi load();
};

// Throws DeviceError when glGetError reports a pending error.
void checkGl(GlApi const &gl, char const *operation);

} // namespace tc::opengl
