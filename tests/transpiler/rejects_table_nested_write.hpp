// expect: a table element can only be written by a statement of its own
#include <tc/tc.hpp>

struct Spark {
    tc::vec3 position;
    float mass;
};

struct TC_KERNEL TableNestedWrite {
    static constexpr char fileLocation[] = "table_nested_write";
    tc::uvec3 local_size{1, 1, 1};
    tc::TableBinding<Spark, 0> sparks;
    tc::BufferBinding<float, 1> out;

    void main() { out[0] = (sparks[0].mass = 2.0f); }
};
