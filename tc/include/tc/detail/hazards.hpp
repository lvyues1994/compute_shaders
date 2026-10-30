#pragma once

#include <cstddef>
#include <vector>

namespace tc::detail {

// One resource access of a device command. `size` == 0 covers the whole
// resource; images are tracked whole.
struct ResourceAccess {
    void const *key = nullptr;
    std::size_t offset = 0;
    std::size_t size = 0;
    bool read = false;
    bool write = false;
};

// Decides where a command stream needs memory barriers: before a command
// that reads or writes what an earlier command wrote, or writes what an
// earlier command read, since the last barrier. Commands that touch
// disjoint resources (or disjoint ranges of one buffer) run without one.
struct HazardTracker {
    // True when `accesses` conflict with the accesses since the last
    // barrier: the caller records a barrier before the command. Either way
    // the command's accesses are remembered.
    bool needsBarrier(std::vector<ResourceAccess> const &accesses) {
        auto conflict = false;
        for (auto const &access : accesses) {
            conflict = conflict or (overlaps(written, access) and (access.read or access.write))
                    or (access.write and overlaps(read, access));
        }
        if (conflict) {
            written.clear();
            read.clear();
        }
        for (auto const &access : accesses) {
            if (access.write) {
                written.push_back(access);
            }
            if (access.read) {
                read.push_back(access);
            }
        }
        return conflict;
    }

    // A barrier the caller recorded for another reason.
    void barrierRecorded() noexcept {
        written.clear();
        read.clear();
    }

private:
    static bool overlaps(std::vector<ResourceAccess> const &earlier, ResourceAccess const &access) noexcept {
        for (auto const &other : earlier) {
            if (other.key != access.key) {
                continue;
            }
            if (other.size == 0 or access.size == 0) {
                return true;
            }
            if (other.offset < access.offset + access.size and access.offset < other.offset + other.size) {
                return true;
            }
        }
        return false;
    }

    std::vector<ResourceAccess> written;
    std::vector<ResourceAccess> read;
};

} // namespace tc::detail
