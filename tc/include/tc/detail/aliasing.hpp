#pragma once

#include <algorithm>
#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

#include <tc/detail/hazards.hpp>

namespace tc::detail {

// Which resources live in heaps (Device::createDeviceBuffer / createImage
// with a Placement), and which of the aliases of each byte was used last.
// Backends key their hazard tracking by heap bytes through access(), so
// aliases are ordered like one resource, and ask claim() before a command
// uses a placed resource: true means its contents are now undefined (it
// was not the last user of its bytes), which Vulkan images answer with a
// layout transition and the CPU backend by filling the bytes with garbage.
struct AliasTracker {
    struct Range {
        void const *heap = nullptr;
        std::size_t offset = 0;
        std::size_t bytes = 0;
    };

    void place(void const *const key, std::weak_ptr<void const> owner, Range const range) {
        purge();
        placed[key] = Placed{std::move(owner), range};
    }

    // nullptr for resources with memory of their own (and for the stale
    // entry of a destroyed resource whose key a new one reuses).
    Range const *rangeOf(void const *const key) const {
        auto const it = placed.find(key);
        return it == placed.end() or it->second.owner.expired() ? nullptr : &it->second.range;
    }

    // The access of `key`'s bytes [offset, offset + size) (size 0: all of
    // them), in heap bytes for placed resources.
    ResourceAccess access(void const *const key, std::size_t const offset, std::size_t const size, bool const read,
                          bool const write) const {
        auto const *const range = rangeOf(key);
        if (range == nullptr) {
            return ResourceAccess{key, offset, size, read, write};
        }
        return ResourceAccess{range->heap, range->offset + offset, size == 0 ? range->bytes : size, read, write};
    }

    // Makes `key` the last user of its bytes; true when it was not already.
    bool claim(void const *const key) {
        auto const *const range = rangeOf(key);
        if (range == nullptr) {
            return false;
        }
        auto &users = lastUsers[range->heap];
        for (auto const &user : users) {
            if (user.key == key and user.offset == range->offset and user.bytes == range->bytes) {
                return false;
            }
        }
        auto const end = range->offset + range->bytes;
        users.erase(std::remove_if(users.begin(), users.end(),
                                   [&](User const &user) {
                                       return user.offset < end and range->offset < user.offset + user.bytes;
                                   }),
                    users.end());
        users.push_back(User{key, range->offset, range->bytes});
        return true;
    }

private:
    struct Placed {
        std::weak_ptr<void const> owner;
        Range range;
    };
    struct User {
        void const *key = nullptr;
        std::size_t offset = 0;
        std::size_t bytes = 0;
    };

    // Forgets destroyed resources, whose keys a new resource may reuse.
    void purge() {
        for (auto it = placed.begin(); it != placed.end();) {
            if (not it->second.owner.expired()) {
                ++it;
                continue;
            }
            auto &users = lastUsers[it->second.range.heap];
            auto const key = it->first;
            users.erase(std::remove_if(users.begin(), users.end(), [key](User const &user) { return user.key == key; }),
                        users.end());
            it = placed.erase(it);
        }
    }

    std::unordered_map<void const *, Placed> placed;
    std::unordered_map<void const *, std::vector<User>> lastUsers;
};

} // namespace tc::detail
