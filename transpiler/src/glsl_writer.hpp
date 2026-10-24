#pragma once

#include <optional>

#include <clang/AST/ASTContext.h>

#include "diagnostics.hpp"
#include "kernel_model.hpp"

namespace tc::transpiler {

// Translates main() and everything it reaches (helper member functions, free
// functions, user structs, constants) into dialect-neutral GLSL. Every C++
// implicit conversion becomes an explicit GLSL constructor, so the output is
// also valid for OpenGL ES, which has no implicit conversions. Records how
// each binding is accessed in `kernel`. Returns nullopt after errors.
std::optional<GlslBody> writeGlsl(KernelModel &kernel, clang::ASTContext &context, Diagnostics &diags);

} // namespace tc::transpiler
