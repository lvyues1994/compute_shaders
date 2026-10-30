// expect: must declare 'static constexpr bool usesSubgroups = true;'
#include <tc/tc.hpp>

using namespace tc;

struct TC_KERNEL SumWithoutDeclaring {
    static constexpr char fileLocation[] = "sum_without_declaring";
    uvec3 local_size{64, 1, 1};

    BufferBinding<uint, 0> values;

    void main() { values[gl_GlobalInvocationID.x] = subgroupAdd(values[gl_GlobalInvocationID.x]); }
};
