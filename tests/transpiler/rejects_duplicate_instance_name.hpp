// expect: program name 'shade' is used by two kernels
#include <tc/tc.hpp>

namespace tests {

using namespace tc;

template <int Scale>
struct TC_KERNEL Shade {
    static constexpr char const *fileLocation = tc::instanceName<Shade>();
    uvec3 local_size{32, 1, 1};
    BufferBinding<float, 0> out;

    void main() { out[gl_GlobalInvocationID.x] = float(Scale); }
};

TC_KERNEL_INSTANCE("shade", Shade<1>);
TC_KERNEL_INSTANCE("shade", Shade<2>);

} // namespace tests
