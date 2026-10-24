// expect: 'new' is not allowed in kernel, only value types are allowed
#include <tc/tc.hpp>

struct TC_KERNEL UsesNew {
    static constexpr char fileLocation[] = "uses_new";
    tc::uvec3 local_size{1, 1, 1};
    tc::BufferBinding<float, 0> out;

    void main() {
        float *value = new float{1.0f};
        out[0] = *value;
        delete value;
    }
};
