#pragma once

#include <clang/AST/ASTContext.h>

#include "diagnostics.hpp"
#include "kernel_model.hpp"

namespace tc::transpiler {

// Checks that every buffer element type has the same byte layout in C++ as
// in a GLSL std430 block: member offsets, struct sizes and array strides.
// A mismatch would silently scramble data between host and GPU.
bool checkBufferLayouts(KernelModel const &kernel, clang::ASTContext &context, Diagnostics &diags);

} // namespace tc::transpiler
