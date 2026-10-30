// expect: table elements are structs, vectors or matrices
#include <tc/tc.hpp>

struct TC_KERNEL TableScalar {
    static constexpr char fileLocation[] = "table_scalar";
    tc::uvec3 local_size{1, 1, 1};
    tc::TableBinding<float, 0> values;

    void main() {}
};
