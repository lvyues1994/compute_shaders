// expect: the index of a table element that is written must not have side effects
#include <tc/tc.hpp>

struct Spark {
    tc::vec3 position;
    float mass;
};

struct TC_KERNEL TableSideEffectIndex {
    static constexpr char fileLocation[] = "table_side_effect_index";
    tc::uvec3 local_size{1, 1, 1};
    tc::TableBinding<Spark, 0> sparks;

    void main() {
        tc::uint i = 0u;
        sparks[i++].position.x += 1.0f;
    }
};
