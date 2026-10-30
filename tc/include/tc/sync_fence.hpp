#pragma once

#include <cerrno>
#include <utility>

#include <poll.h>
#include <unistd.h>

namespace tc {

// A sync file descriptor (Android sync file, Linux sync_file): it becomes
// readable once the work it stands for has finished. Owns the descriptor;
// an empty fence has nothing to wait for.
struct SyncFence {
    SyncFence() noexcept = default;
    explicit SyncFence(int const fd_) noexcept : fd(fd_) {}
    SyncFence(SyncFence &&other) noexcept : fd(std::exchange(other.fd, -1)) {}
    SyncFence &operator=(SyncFence &&other) noexcept {
        if (this != &other) {
            reset();
            fd = std::exchange(other.fd, -1);
        }
        return *this;
    }
    SyncFence(SyncFence const &) = delete;
    SyncFence &operator=(SyncFence const &) = delete;
    ~SyncFence() { reset(); }

    bool empty() const noexcept { return fd < 0; }
    int get() const noexcept { return fd; }
    // Gives up ownership, e.g. to ASurfaceTransaction_setBuffer.
    int release() noexcept { return std::exchange(fd, -1); }

    // Whether the work has finished; true for an empty fence.
    bool isSignaled() const noexcept { return poll(0); }
    // Blocks the calling thread until the work has finished.
    void wait() const noexcept { poll(-1); }

private:
    bool poll(int const timeout) const noexcept {
        if (fd < 0) {
            return true;
        }
        auto descriptor = pollfd{fd, POLLIN, 0};
        for (;;) {
            auto const ready = ::poll(&descriptor, 1, timeout);
            if (ready >= 0) {
                return ready > 0;
            }
            if (errno != EINTR and errno != EAGAIN) {
                return true; // an invalid descriptor has nothing left to wait for
            }
        }
    }

    void reset() noexcept {
        if (fd >= 0) {
            ::close(fd);
        }
        fd = -1;
    }

    int fd = -1;
};

} // namespace tc
