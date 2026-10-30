#pragma once

#include <optional>
#include <vector>

#include <clang/AST/ASTContext.h>

#include "diagnostics.hpp"
#include "kernel_model.hpp"

namespace tc::transpiler {

// tc::Encoded<V, Codec>: V in GLSL, Codec::words words in a table, packed
// and unpacked by the transpiled Codec::encode / Codec::decode.
struct EncodedInfo {
    clang::QualType value;
    std::uint32_t words = 0;
    clang::CXXMethodDecl const *encode = nullptr;
    clang::CXXMethodDecl const *decode = nullptr;
};

// nullopt when `type` is not an Encoded specialisation; reports malformed
// codecs at `loc`.
std::optional<EncodedInfo> encodedInfo(clang::QualType type, clang::ASTContext &context, Diagnostics &diags,
                                       clang::SourceLocation loc);

// Words a value of `type` takes in a table: the rules of tc::detail::Packing
// (32-bit scalars take one word, double two; vectors, matrices, std::array
// and structs concatenate their parts without padding). Reports unsupported
// types at `loc`.
std::optional<std::uint32_t> packedWords(clang::QualType type, clang::ASTContext &context, Diagnostics &diags,
                                         clang::SourceLocation loc);

// The fields of a table element: those of a struct, or the element itself.
std::optional<PackedLayout> packedLayout(clang::QualType element, clang::ASTContext &context, Diagnostics &diags,
                                         clang::SourceLocation loc);

// The functions of a table layout type; reports what is missing at `loc`.
std::optional<TableAddressing> tableAddressing(clang::QualType layout, Diagnostics &diags, clang::SourceLocation loc);

// Calls a constexpr function with unsigned integer arguments at build time,
// as Clang's constant evaluator does for constant expressions.
std::optional<std::uint64_t> evaluateCall(clang::ASTContext &context, clang::FunctionDecl const *function,
                                          std::vector<std::uint64_t> const &arguments);

} // namespace tc::transpiler
