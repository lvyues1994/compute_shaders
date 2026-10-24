#include "tc_types.hpp"

#include <clang/AST/DeclBase.h>

namespace tc::transpiler {

llvm::StringRef rootNamespace(clang::Decl const *decl) {
    auto root = llvm::StringRef{};
    for (auto const *ctx = decl->getDeclContext(); ctx != nullptr; ctx = ctx->getParent()) {
        if (auto const *ns = llvm::dyn_cast<clang::NamespaceDecl>(ctx)) {
            if (not ns->isInline() and not ns->isAnonymousNamespace()) {
                root = ns->getName();
            }
        }
    }
    return root;
}

bool isTcDecl(clang::Decl const *decl) {
    return decl != nullptr and rootNamespace(decl) == "tc";
}

clang::QualType bareType(clang::QualType const type) {
    return type.getNonReferenceType().getCanonicalType().getUnqualifiedType();
}

clang::ClassTemplateSpecializationDecl const *asTcSpecialization(clang::RecordDecl const *record,
                                                                 llvm::StringRef const name) {
    auto const *spec = llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(record);
    if (spec == nullptr) {
        return nullptr;
    }
    auto const *tmpl = spec->getSpecializedTemplate();
    if (tmpl->getQualifiedNameAsString() != ("tc::" + name).str()) {
        return nullptr;
    }
    return spec;
}

clang::ClassTemplateSpecializationDecl const *asTcSpecialization(clang::QualType const type,
                                                                 llvm::StringRef const name) {
    return asTcSpecialization(bareType(type)->getAsCXXRecordDecl(), name);
}

std::optional<StdArrayInfo> asStdArray(clang::QualType const type) {
    auto const *spec =
        llvm::dyn_cast_or_null<clang::ClassTemplateSpecializationDecl>(bareType(type)->getAsCXXRecordDecl());
    if (spec == nullptr or not spec->isInStdNamespace() or spec->getName() != "array") {
        return std::nullopt;
    }
    return StdArrayInfo{templateTypeArg(spec, 0), templateIntArg(spec, 1)};
}

std::optional<VecInfo> asTcVec(clang::QualType const type) {
    auto const *spec = asTcSpecialization(type, "vec");
    if (spec == nullptr) {
        return std::nullopt;
    }
    return VecInfo{templateTypeArg(spec, 0), static_cast<unsigned>(templateIntArg(spec, 1))};
}

std::optional<MatInfo> asTcMat(clang::QualType const type) {
    auto const *spec = asTcSpecialization(type, "mat");
    if (spec == nullptr) {
        return std::nullopt;
    }
    return MatInfo{templateTypeArg(spec, 0), static_cast<unsigned>(templateIntArg(spec, 1)),
                   static_cast<unsigned>(templateIntArg(spec, 2))};
}

clang::QualType templateTypeArg(clang::ClassTemplateSpecializationDecl const *spec, unsigned const index) {
    return spec->getTemplateArgs()[index].getAsType();
}

std::uint64_t templateIntArg(clang::ClassTemplateSpecializationDecl const *spec, unsigned const index) {
    return spec->getTemplateArgs()[index].getAsIntegral().getZExtValue();
}

} // namespace tc::transpiler
