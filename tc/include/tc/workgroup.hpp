#pragma once

#include <cstdint>

// Workgroup synchronisation: GLSL barrier() and the memory barriers.
//
// A kernel that calls barrier() declares `static constexpr bool usesBarrier
// = true;`. The CPU backend then runs each invocation of a workgroup as a
// fiber and switches fibers at barrier(), so invocations really wait for each
// other. It aborts with a message when some invocations finish while others
// wait at a barrier (undefined behaviour on GPUs).
namespace tc {

namespace detail {

// Callbacks for one workgroup of a usesBarrier kernel.
struct CooperativeGroup {
    void *context = nullptr;
    void (*prepare)(void *context, std::uint32_t invocation) = nullptr; // set built-ins before (re)entering
    void (*run)(void *context, std::uint32_t invocation) = nullptr;     // the invocation's main()
};

// Runs `count` invocations so that they can wait for each other in barrier().
void runCooperative(std::uint32_t count, CooperativeGroup const &group);

void workgroupBarrier();

} // namespace detail

inline void barrier() {
    detail::workgroupBarrier();
}

// Invocations of one CPU workgroup share a thread, so their memory accesses
// are already ordered; other workgroups are not synchronised, as on GPUs.
inline void memoryBarrier() noexcept {}
inline void memoryBarrierShared() noexcept {}
inline void memoryBarrierBuffer() noexcept {}
inline void memoryBarrierImage() noexcept {}
inline void groupMemoryBarrier() noexcept {}

} // namespace tc
