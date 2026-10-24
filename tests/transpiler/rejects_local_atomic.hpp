// expect: atomicAdd needs a buffer element or a Shared variable as its first argument
#include <tc/tc.hpp>

struct TC_KERNEL LocalAtomic {
    static constexpr char fileLocation[] = "local_atomic";
    tc::uvec3 local_size{1, 1, 1};
    tc::BufferBinding<tc::uint, 0> out;

    void main() {
        tc::uint counter = 0u;
        tc::atomicAdd(counter, 1u);
        out[0] = counter;
    }
};
