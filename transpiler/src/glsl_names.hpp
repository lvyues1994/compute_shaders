#pragma once

#include <optional>
#include <string>

#include <llvm/ADT/StringRef.h>

namespace tc::transpiler {

// A C++ identifier made safe for GLSL: names that are GLSL keywords,
// built-in function names or reserved prefixes get a suffix.
std::string glslIdentifier(llvm::StringRef name);

// GLSL name of a <cmath> / std:: function allowed in kernels, e.g.
// std::sqrt -> sqrt, fabsf -> abs, std::atan2 -> atan.
std::optional<std::string> glslNameOfStdFunction(llvm::StringRef name);

// GLSL name of a tc:: built-in function (tc::not_ -> not).
std::string glslNameOfTcFunction(llvm::StringRef name);

bool isSwizzleName(llvm::StringRef name);

} // namespace tc::transpiler
