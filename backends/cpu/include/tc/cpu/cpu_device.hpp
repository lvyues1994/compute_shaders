#pragma once

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include <tc/device.hpp>

namespace tc::cpu {

// A non-owning reference to a callable `void(std::uint32_t)`.
struct TaskRef {
    template <class F, std::enable_if_t<not std::is_same_v<std::decay_t<F>, TaskRef>, int> = 0>
    TaskRef(F &&f) noexcept // NOLINT: implicit, like std::function_ref
        : context(const_cast<void *>(static_cast<void const *>(&f))),
          call([](void *const c, std::uint32_t const index) { (*static_cast<std::remove_reference_t<F> *>(c))(index); }) {}

    void operator()(std::uint32_t const index) const { call(context, index); }

private:
    void *context;
    void (*call)(void *, std::uint32_t);
};

// How the CPU backend spreads the workgroups of a dispatch over threads.
// Engines implement it on their own pool so kernels share its threads; tc
// does not depend on any particular pool.
struct ParallelExecutor {
    virtual ~ParallelExecutor() = default;

    // Tasks that can run at the same time.
    virtual std::uint32_t concurrency() const = 0;

    // Calls task(i) for every i in [0, count) and returns when all calls
    // have returned. Requirements:
    // - Each call runs from start to end on one thread, and that thread
    //   runs nothing else in between: workgroups keep thread-local state,
    //   and barrier() switches fibers on that thread.
    // - It is called from the thread that calls Device::execute(), which
    //   may be one of the executor's own workers. It must then make progress
    //   by running tasks on the calling thread (or by idle workers taking
    //   them) rather than block, or a busy pool deadlocks.
    virtual void bulk(std::uint32_t count, TaskRef task) = 0;
};

// tc's own executor: runs `threads` tasks at once (0: one per hardware
// thread) on threads - 1 persistent workers plus the thread calling bulk().
std::shared_ptr<ParallelExecutor> makeThreadPool(std::uint32_t threads = 0);

struct CpuDeviceConfig {
    std::uint32_t threadCount = 0; // 0 selects std::thread::hardware_concurrency(); ignored with `executor`
    // Runs the workgroups; nullptr: a thread pool of `threadCount` threads
    // owned by the device.
    std::shared_ptr<ParallelExecutor> executor;
    // Invocations per subgroup: a power of two from 1 to 128.
    std::uint32_t subgroupSize = 32;
};

// Runs kernels as plain C++: workgroups spread over the executor's threads,
// the invocations of one workgroup run sequentially. Kernels can be
// debugged with a normal debugger and sanitizers.
std::unique_ptr<Device> createDevice(CpuDeviceConfig const &config = {});

} // namespace tc::cpu
