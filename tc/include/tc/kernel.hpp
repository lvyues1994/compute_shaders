#pragma once

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

} // namespace tc
