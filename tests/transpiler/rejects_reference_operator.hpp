// expect: operators such as \+= that return \*this cannot be translated
#include <tc/tc.hpp>

struct Accumulator {
    float total;

    Accumulator &operator+=(float const value) {
        total += value;
        return *this;
    }
};

struct TC_KERNEL ReferenceOperator {
    static constexpr char fileLocation[] = "reference_operator";
    tc::uvec3 local_size{1, 1, 1};
    tc::BufferBinding<float, 0> out;

    void main() {
        Accumulator sum = Accumulator{0.0f};
        sum += 2.0f;
        out[0] = sum.total;
    }
};
