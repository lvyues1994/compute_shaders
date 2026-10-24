// expect: kernel struct needs a 'void main\(\)' member function
#include <tc/tc.hpp>

struct TC_KERNEL NoEntryPoint {
    static constexpr char fileLocation[] = "no_entry_point";
    tc::uvec3 local_size{1, 1, 1};
    tc::BufferBinding<int, 0> out;
};
