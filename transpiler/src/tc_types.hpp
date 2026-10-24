#pragma once

#include <cstdint>
#include <optional>

#include <clang/AST/ASTContext.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/DeclTemplate.h>
#include <clang/AST/Type.h>
#include <llvm/ADT/StringRef.h>

// Recognises the tc DSL types and the few standard types kernels may use.
namespace tc::transpiler {

// The outermost named namespace of a declaration ("tc", "std", "").
llvm::StringRef rootNamespace(clang::Decl const *decl);

// Is `decl` declared inside namespace tc (at any depth)?
bool isTcDecl(clang::Decl const *decl);

// Specialisation of tc::<name> (e.g. "vec", "BufferBinding", "detail::VecFields").
clang::ClassTemplateSpecializationDecl const *asTcSpecialization(clang::QualType type, llvm::StringRef name);
clang::ClassTemplateSpecializationDecl const *asTcSpecialization(clang::RecordDecl const *record,
                                                                 llvm::StringRef name);

// std::array<T, N>.
struct StdArrayInfo {
    clang::QualType element;
    std::uint64_t size = 0;
};
std::optional<StdArrayInfo> asStdArray(clang::QualType type);

// tc::vec<T, N>.
struct VecInfo {
    clang::QualType element;
    unsigned size = 0;
};
std::optional<VecInfo> asTcVec(clang::QualType type);

clang::QualType templateTypeArg(clang::ClassTemplateSpecializationDecl const *spec, unsigned index);
std::uint64_t templateIntArg(clang::ClassTemplateSpecializationDecl const *spec, unsigned index);

// The canonical, unqualified, non-reference form of a type.
clang::QualType bareType(clang::QualType type);

} // namespace tc::transpiler
