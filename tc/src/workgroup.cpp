#include <tc/workgroup.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#if (defined(__x86_64__) or defined(__aarch64__)) and defined(__linux__)
#define TC_FIBERS 1
#include <sys/mman.h>
#include <unistd.h>
#else
#define TC_FIBERS 0
#include <condition_variable>
#include <mutex>
#include <thread>
#endif

namespace tc::detail {

namespace {

[[noreturn]] void fail(char const *message) {
    std::fprintf(stderr, "tc: %s\n", message);
    std::abort();
}

constexpr char kNotCooperative[] =
    "barrier() called by a kernel that does not declare `static constexpr bool usesBarrier = true`";
constexpr char kDiverged[] =
    "barrier() was not reached by every invocation of a workgroup (some returned or branched around it); "
    "on a GPU this hangs or corrupts shared memory";

} // namespace

#if TC_FIBERS

// Saves the callee-saved registers on the current stack, stores the stack
// pointer in *save, switches to `load` and restores the registers saved there.
extern "C" void tc_fiber_switch(void **save, void *load);
// First code a new fiber runs: calls entry(fiber) from the initial frame.
extern "C" void tc_fiber_trampoline();

#if defined(__x86_64__)
asm(R"(
    .text
    .globl tc_fiber_switch
    .type tc_fiber_switch, @function
    .p2align 4
tc_fiber_switch:
    pushq %rbp
    pushq %rbx
    pushq %r12
    pushq %r13
    pushq %r14
    pushq %r15
    subq $16, %rsp
    stmxcsr 8(%rsp)
    fnstcw (%rsp)
    movq %rsp, (%rdi)
    movq %rsi, %rsp
    fldcw (%rsp)
    ldmxcsr 8(%rsp)
    addq $16, %rsp
    popq %r15
    popq %r14
    popq %r13
    popq %r12
    popq %rbx
    popq %rbp
    ret
    .size tc_fiber_switch, .-tc_fiber_switch

    .globl tc_fiber_trampoline
    .type tc_fiber_trampoline, @function
    .p2align 4
tc_fiber_trampoline:
    movq %r12, %rdi
    callq *%r13
    ud2
    .size tc_fiber_trampoline, .-tc_fiber_trampoline
)");
#else
asm(R"(
    .text
    .globl tc_fiber_switch
    .type tc_fiber_switch, %function
    .p2align 4
tc_fiber_switch:
    sub sp, sp, #160
    stp x19, x20, [sp, #0]
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]
    stp x27, x28, [sp, #64]
    stp x29, x30, [sp, #80]
    stp d8, d9, [sp, #96]
    stp d10, d11, [sp, #112]
    stp d12, d13, [sp, #128]
    stp d14, d15, [sp, #144]
    mov x2, sp
    str x2, [x0]
    mov sp, x1
    ldp x19, x20, [sp, #0]
    ldp x21, x22, [sp, #16]
    ldp x23, x24, [sp, #32]
    ldp x25, x26, [sp, #48]
    ldp x27, x28, [sp, #64]
    ldp x29, x30, [sp, #80]
    ldp d8, d9, [sp, #96]
    ldp d10, d11, [sp, #112]
    ldp d12, d13, [sp, #128]
    ldp d14, d15, [sp, #144]
    add sp, sp, #160
    ret
    .size tc_fiber_switch, .-tc_fiber_switch

    .globl tc_fiber_trampoline
    .type tc_fiber_trampoline, %function
    .p2align 4
tc_fiber_trampoline:
    mov x0, x19
    blr x20
    brk #0
    .size tc_fiber_trampoline, .-tc_fiber_trampoline
)");
#endif

namespace {

constexpr std::size_t kStackSize = 64 * 1024;

// A fiber stack with an inaccessible guard page below it, so an overflow
// faults instead of corrupting the neighbouring stack.
struct FiberStack {
    FiberStack() : guard(static_cast<std::size_t>(sysconf(_SC_PAGESIZE))) {
        auto *const memory = mmap(nullptr, guard + kStackSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (memory == MAP_FAILED) {
            fail("cannot allocate a fiber stack");
        }
        base = static_cast<std::byte *>(memory);
        mprotect(base, guard, PROT_NONE);
    }
    FiberStack(FiberStack &&other) noexcept : base(std::exchange(other.base, nullptr)), guard(other.guard) {}
    FiberStack &operator=(FiberStack &&) = delete;
    FiberStack(FiberStack const &) = delete;
    FiberStack &operator=(FiberStack const &) = delete;
    ~FiberStack() {
        if (base != nullptr) {
            munmap(base, guard + kStackSize);
        }
    }

    std::byte *top() const noexcept { return base + guard + kStackSize; }

private:
    std::byte *base = nullptr;
    std::size_t guard;
};

struct Scheduler;

struct Fiber {
    void *sp = nullptr;
    bool done = false;
    std::uint32_t index = 0;
    Scheduler *scheduler = nullptr;
};

struct Scheduler {
    void *sp = nullptr;
    Fiber *current = nullptr;
    CooperativeGroup const *group = nullptr;
};

thread_local Scheduler *activeScheduler = nullptr;
thread_local std::vector<FiberStack> stackPool; // reused by later workgroups on this thread

void fiberMain(Fiber *const fiber) {
    auto *const scheduler = fiber->scheduler;
    scheduler->group->run(scheduler->group->context, fiber->index);
    fiber->done = true;
    tc_fiber_switch(&fiber->sp, scheduler->sp);
    fail("a finished fiber was resumed");
}

// The frame tc_fiber_switch restores when it first switches to `fiber`: its
// return address is the trampoline, which calls fiberMain(fiber).
void *initialFrame(std::byte *const top, Fiber *const fiber) {
    auto const topAddress = reinterpret_cast<std::uintptr_t>(top);
#if defined(__x86_64__)
    // 16 bytes of x87/SSE control words, r15..rbp, return address; the
    // trampoline must start with a 16-byte aligned stack pointer.
    auto *const frame = reinterpret_cast<std::uint64_t *>(((topAddress - 80) & ~std::uintptr_t{15}) + 8);
    frame[0] = 0x037F;                                                 // x87 control word
    frame[1] = 0x1F80;                                                 // MXCSR
    frame[2] = 0;                                                      // r15
    frame[3] = 0;                                                      // r14
    frame[4] = reinterpret_cast<std::uint64_t>(&fiberMain);           // r13: entry
    frame[5] = reinterpret_cast<std::uint64_t>(fiber);                // r12: argument
    frame[6] = 0;                                                      // rbx
    frame[7] = 0;                                                      // rbp
    frame[8] = reinterpret_cast<std::uint64_t>(&tc_fiber_trampoline); // return address
#else
    // x19..x30 then d8..d15, 160 bytes, 16-byte aligned.
    auto *const frame = reinterpret_cast<std::uint64_t *>((topAddress - 160) & ~std::uintptr_t{15});
    for (int i = 0; i < 20; ++i) {
        frame[i] = 0;
    }
    frame[0] = reinterpret_cast<std::uint64_t>(fiber);                 // x19: argument
    frame[1] = reinterpret_cast<std::uint64_t>(&fiberMain);            // x20: entry
    frame[11] = reinterpret_cast<std::uint64_t>(&tc_fiber_trampoline); // x30: return address
#endif
    return frame;
}

} // namespace

void runCooperative(std::uint32_t const count, CooperativeGroup const &group) {
    if (activeScheduler != nullptr) {
        fail("cooperative workgroups cannot nest");
    }
    while (stackPool.size() < count) {
        stackPool.emplace_back();
    }
    auto scheduler = Scheduler{};
    scheduler.group = &group;
    auto fibers = std::vector<Fiber>(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        fibers[i].index = i;
        fibers[i].scheduler = &scheduler;
        fibers[i].sp = initialFrame(stackPool[i].top(), &fibers[i]);
    }
    activeScheduler = &scheduler;
    // Each round resumes every unfinished invocation until its next barrier
    // or its end; a barrier is complete when the round has visited them all.
    for (auto remaining = count; remaining > 0;) {
        auto finished = 0u;
        auto waiting = 0u;
        for (auto &fiber : fibers) {
            if (fiber.done) {
                continue;
            }
            group.prepare(group.context, fiber.index);
            scheduler.current = &fiber;
            tc_fiber_switch(&scheduler.sp, fiber.sp);
            if (fiber.done) {
                ++finished;
                --remaining;
            } else {
                ++waiting;
            }
        }
        if (finished != 0 and waiting != 0) {
            fail(kDiverged);
        }
    }
    activeScheduler = nullptr;
}

void workgroupBarrier() {
    auto *const scheduler = activeScheduler;
    if (scheduler == nullptr) {
        fail(kNotCooperative);
    }
    auto *const fiber = scheduler->current;
    tc_fiber_switch(&fiber->sp, scheduler->sp);
}

#else

namespace {

// Portable fallback: one thread per invocation and a counting barrier.
struct ThreadGroup {
    explicit ThreadGroup(std::uint32_t const count_) : count(count_) {}

    std::mutex mutex;
    std::condition_variable wake;
    std::uint32_t count;
    std::uint32_t arrived = 0;
    std::uint32_t finished = 0;
    std::uint64_t generation = 0;
};

thread_local ThreadGroup *activeGroup = nullptr;

} // namespace

void runCooperative(std::uint32_t const count, CooperativeGroup const &group) {
    auto shared = ThreadGroup{count};
    auto threads = std::vector<std::thread>{};
    threads.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        threads.emplace_back([&shared, &group, i] {
            activeGroup = &shared;
            group.prepare(group.context, i);
            group.run(group.context, i);
            auto const lock = std::lock_guard<std::mutex>{shared.mutex};
            ++shared.finished;
            if (shared.arrived != 0 and shared.arrived + shared.finished == shared.count) {
                fail(kDiverged);
            }
        });
    }
    for (auto &thread : threads) {
        thread.join();
    }
}

void workgroupBarrier() {
    auto *const group = activeGroup;
    if (group == nullptr) {
        fail(kNotCooperative);
    }
    auto lock = std::unique_lock<std::mutex>{group->mutex};
    auto const generation = group->generation;
    if (++group->arrived == group->count) {
        group->arrived = 0;
        ++group->generation;
        group->wake.notify_all();
        return;
    }
    if (group->arrived + group->finished == group->count) {
        fail(kDiverged);
    }
    group->wake.wait(lock, [group, generation] { return group->generation != generation; });
}

#endif

} // namespace tc::detail
