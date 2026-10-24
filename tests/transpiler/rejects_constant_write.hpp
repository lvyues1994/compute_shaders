// expect: kernel member 'counter' is a constant on the GPU and cannot be modified
#include <tc/tc.hpp>

struct TC_KERNEL WritesMember {
    static constexpr char fileLocation[] = "writes_member";
    tc::uvec3 local_size{1, 1, 1};
    tc::BufferBinding<int, 0> out;
    int counter = 0;

    void main() {
        counter += 1;
        out[0] = counter;
    }
};
