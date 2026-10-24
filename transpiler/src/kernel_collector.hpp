#pragma once

#include <optional>
#include <vector>

#include <clang/AST/ASTContext.h>
#include <clang/AST/DeclCXX.h>

#include "diagnostics.hpp"
#include "kernel_model.hpp"

namespace tc::transpiler {

// Kernel structs defined in the main file: definitions carrying
// [[clang::annotate("kernel")]], found with AST matchers.
std::vector<clang::CXXRecordDecl const *> findKernels(clang::ASTContext &context, Diagnostics &diags);

// Classifies the members of a kernel struct into bindings, uniforms and the
// launch configuration. Returns nullopt after reporting errors.
std::optional<KernelModel> collectKernel(clang::CXXRecordDecl const *record, clang::ASTContext &context,
                                         Diagnostics &diags);

} // namespace tc::transpiler
