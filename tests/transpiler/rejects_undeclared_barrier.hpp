// expect: kernels that call barrier\(\) must declare 'static constexpr bool usesBarrier = true;'
#include <tc/tc.hpp>

struct TC_KERNEL UndeclaredBarrier {
    static constexpr char fileLocation[] = "undeclared_barrier";
    tc::uvec3 local_size{64, 1, 1};
    tc::BufferBinding<tc::uint, 0> out;
    tc::Shared<std::array<tc::uint, 64>> scratch;

    void main() {
        tc::uint i = tc::gl_LocalInvocationIndex;
        scratch[i] = i;
        tc::barrier();
        out[i] = scratch[63u - i];
    }
};
