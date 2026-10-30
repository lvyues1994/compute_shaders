// expect: codec 'Truncate' needs 'static constexpr std::uint32_t words'
#include <tc/tc.hpp>

struct Truncate {
    using Value = float;
    static constexpr std::uint32_t words = 1;
    static std::array<tc::uint, 1> encode(float const &value) { return std::array<tc::uint, 1>{tc::uint(value)}; }
};

struct Sample {
    tc::Encoded<float, Truncate> value;
    float weight;
};

struct TC_KERNEL CodecWithoutDecode {
    static constexpr char fileLocation[] = "codec_without_decode";
    tc::uvec3 local_size{1, 1, 1};
    tc::TableBinding<Sample, 0> samples;

    void main() {}
};
