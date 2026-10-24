// expect: 'printf' is not supported in GPU kernels
#include <cstdio>

#include <tc/tc.hpp>

struct TC_KERNEL CallsPrintf {
    static constexpr char fileLocation[] = "calls_printf";
    tc::uvec3 local_size{1, 1, 1};
    tc::BufferBinding<int, 0> out;

    void main() { out[0] = std::printf("hello"); }
};
