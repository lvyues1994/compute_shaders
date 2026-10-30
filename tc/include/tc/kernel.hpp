#pragma once

#include <cstdint>
#include <type_traits>
#include <utility>

#include <tc/vec.hpp>

// Marks a struct as a compute kernel. The transpiler finds kernels through
// this Clang annotation; other compilers just ignore the struct attribute.
#if defined(__clang__)
#define TC_KERNEL [[clang::annotate("kernel")]]
#else
#define TC_KERNEL
#endif

namespace tc {

// A kernel struct provides:
//   static constexpr char fileLocation[]  name of the generated GLSL program
//   uvec3 local_size                      workgroup size
//   void main()                           the per-invocation entry point
template <class K, class = void>
struct IsKernel : std::false_type {};

template <class K>
struct IsKernel<K, std::void_t<decltype(K::fileLocation), decltype(uvec3{std::declval<K const &>().local_size}),
                               decltype(std::declval<K &>().main())>> : std::true_type {};

template <class K>
inline constexpr bool isKernel = IsKernel<K>::value;

// `static constexpr bool usesBarrier = true;` marks kernels whose
// invocations synchronise with barrier().
template <class K, class = void>
struct UsesBarrier : std::false_type {};

template <class K>
struct UsesBarrier<K, std::void_t<decltype(K::usesBarrier)>> : std::bool_constant<K::usesBarrier> {};

// `static constexpr tc::Precision floatPrecision = tc::Precision::Medium;`
// lets OpenGL ES and Vulkan compute the kernel's float arithmetic at medium
// precision (GLSL mediump, SPIR-V RelaxedPrecision): often 16-bit on mobile
// GPUs, with less register pressure and power. Buffers, images, uniforms,
// shared variables and constants keep full precision; desktop OpenGL and
// the CPU backend compute in 32 bits, so results differ by fp16 rounding.
enum class Precision : std::uint8_t {
    High,
    Medium,
};

// Kernel templates. Each instance is its own GLSL program, named where it
// is instantiated:
//
//   template <class Material>
//   struct TC_KERNEL Shade {
//       static constexpr char const *fileLocation = tc::instanceName<Shade>();
//       ...
//   };
//   TC_KERNEL_INSTANCE("shade_lambert", Shade<Lambert>);
//
// TC_KERNEL_INSTANCE goes in the namespace of the kernel template, after
// it; tc-transpile generates exactly the instances named in the header.
template <class K>
struct KernelTag {};

template <class K>
constexpr char const *instanceName() noexcept {
    return tcKernelName(KernelTag<K>{}); // declared by TC_KERNEL_INSTANCE, found through the tag
}

} // namespace tc

#define TC_KERNEL_INSTANCE(name, ...)                                                                                  \
    constexpr char const *tcKernelName(::tc::KernelTag<__VA_ARGS__>) noexcept {                                        \
        return name;                                                                                                   \
    }                                                                                                                  \
    TC_DETAIL_KERNEL_INSTANCE(__VA_ARGS__)

// Only tc-transpile needs the instance's member definitions; host code
// instantiates what it uses. Building the kernel value also instantiates
// its default member initializers (local_size, constants).
#if defined(TC_TRANSPILER)
#define TC_DETAIL_KERNEL_INSTANCE(...)                                                                                 \
    template struct __VA_ARGS__;                                                                                       \
    inline void tcInstantiateKernel(::tc::KernelTag<__VA_ARGS__>) {                                                    \
        (void)__VA_ARGS__{};                                                                                           \
    }                                                                                                                  \
    static_assert(true)
#else
#define TC_DETAIL_KERNEL_INSTANCE(...) static_assert(true)
#endif
