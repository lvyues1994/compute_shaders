// expect: std430 layout mismatch in 'Packed': member 'weight' is at byte 16 in C\+\+ but 12 in GLSL
#include <tc/tc.hpp>

struct Packed {
    tc::vec3 position;
    float weight;
};

struct TC_KERNEL LayoutMismatch {
    static constexpr char fileLocation[] = "layout_mismatch";
    tc::uvec3 local_size{1, 1, 1};
    tc::BufferBinding<Packed, 0> data;

    void main() { data[0].weight = 1.0f; }
};
