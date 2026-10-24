#pragma once

#include <tc/tc.hpp>

namespace examples {

using namespace tc;

// Case 1 of the talk: C[i] = A[i] + B[i], one invocation per element.
struct TC_KERNEL FloatAdder {
    static constexpr char fileLocation[] = "float_adder";
    uvec3 local_size{256, 1, 1};

    BufferBinding<float, 0> A;
    BufferBinding<float, 1> B;
    BufferBinding<float, 2> C;

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= uint(C.length())) {
            return;
        }
        C[i] = A[i] + B[i];
    }
};

} // namespace examples
