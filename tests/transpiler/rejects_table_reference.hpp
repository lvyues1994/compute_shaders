// expect: 'spark' must be a copy of the table element
#include <tc/tc.hpp>

struct Spark {
    tc::vec3 position;
    float mass;
};

struct TC_KERNEL TableReference {
    static constexpr char fileLocation[] = "table_reference";
    tc::uvec3 local_size{1, 1, 1};
    tc::TableBinding<Spark, 0> sparks;

    void main() {
        Spark const &spark = sparks[0];
        sparks[1].mass = spark.mass;
    }
};
