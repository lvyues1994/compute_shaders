// expect: C arrays cannot be stored in tables; use std::array
#include <tc/tc.hpp>

struct Samples {
    float values[4];
};

struct TC_KERNEL TableCArray {
    static constexpr char fileLocation[] = "table_c_array";
    tc::uvec3 local_size{1, 1, 1};
    tc::TableBinding<Samples, 0> samples;

    void main() {}
};
