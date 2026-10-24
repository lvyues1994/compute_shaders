// expect: 'factorial' is called recursively
#include <tc/tc.hpp>

struct TC_KERNEL Recursive {
    static constexpr char fileLocation[] = "recursive";
    tc::uvec3 local_size{1, 1, 1};
    tc::BufferBinding<int, 0> out;

    int factorial(int n) const { return n <= 1 ? 1 : n * factorial(n - 1); }

    void main() { out[0] = factorial(5); }
};
