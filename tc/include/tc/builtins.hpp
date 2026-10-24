#pragma once

#include <tc/vec.hpp>

// GLSL compute-shader input variables. On the GPU they are provided by the
// hardware; the CPU backend assigns them per invocation on each worker
// thread before calling the kernel's main(). Kernels must only read them.
namespace tc {

inline thread_local uvec3 gl_NumWorkGroups{};
inline thread_local uvec3 gl_WorkGroupSize{};
inline thread_local uvec3 gl_WorkGroupID{};
inline thread_local uvec3 gl_LocalInvocationID{};
inline thread_local uvec3 gl_GlobalInvocationID{};
inline thread_local uint gl_LocalInvocationIndex{};

} // namespace tc
