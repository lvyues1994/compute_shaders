#include "packed_layout.hpp"

#include <string>

#include <clang/AST/Expr.h>

#include "tc_types.hpp"

namespace tc::transpiler {

namespace {

// A plain struct the table packs field by field (tc::detail::isRecordSchema).
clang::CXXRecordDecl const *recordSchema(clang::QualType const type) {
    auto const *record = bareType(type)->getAsCXXRecordDecl();
    if (record == nullptr or isTcDecl(record) or asStdArray(type)) {
        return nullptr;
    }
    return record->getDefinition();
}

std::optional<std::uint32_t> recordWords(clang::CXXRecordDecl const *record, clang::QualType const type,
                                         clang::ASTContext &context, Diagnostics &diags,
                                         clang::SourceLocation const loc) {
    if (record->isUnion() or record->getNumBases() != 0 or not record->isAggregate()) {
        diags.error(loc, "table element '" + type.getAsString()
                             + "' must be a plain struct: no base classes, constructors, private members or unions");
        return std::nullopt;
    }
    auto total = std::uint32_t{0};
    auto any = false;
    for (auto const *field : record->fields()) {
        any = true;
        if (field->isBitField()) {
            diags.error(field->getLocation(), "bit-fields cannot be stored in tables");
            return std::nullopt;
        }
        auto const words = packedWords(field->getType(), context, diags, field->getLocation());
        if (not words) {
            return std::nullopt;
        }
        total += *words;
    }
    if (not any) {
        diags.error(loc, "table element '" + type.getAsString() + "' has no fields");
        return std::nullopt;
    }
    return total;
}

clang::CXXMethodDecl const *staticMethod(clang::CXXRecordDecl const *record, llvm::StringRef const name) {
    for (auto const *method : record->methods()) {
        if (method->isStatic() and method->getDeclName().isIdentifier() and method->getName() == name) {
            return method;
        }
    }
    return nullptr;
}

} // namespace

std::optional<EncodedInfo> encodedInfo(clang::QualType const type, clang::ASTContext & /*context*/,
                                       Diagnostics &diags, clang::SourceLocation const loc) {
    auto const *spec = asTcSpecialization(type, "Encoded");
    if (spec == nullptr) {
        return std::nullopt;
    }
    auto info = EncodedInfo{};
    info.value = templateTypeArg(spec, 0);
    auto const codecType = templateTypeArg(spec, 1);
    auto const *codec = bareType(codecType)->getAsCXXRecordDecl();
    codec = codec != nullptr ? codec->getDefinition() : nullptr;
    if (codec == nullptr) {
        diags.error(loc, "codec '" + codecType.getAsString() + "' has no definition");
        return std::nullopt;
    }
    for (auto const *decl : codec->decls()) {
        auto const *var = llvm::dyn_cast<clang::VarDecl>(decl);
        if (var == nullptr or var->getName() != "words" or var->getInit() == nullptr) {
            continue;
        }
        if (auto const *value = var->isConstexpr() ? var->evaluateValue() : nullptr; value != nullptr and value->isInt()) {
            info.words = static_cast<std::uint32_t>(value->getInt().getZExtValue());
        }
    }
    info.encode = staticMethod(codec, "encode");
    info.decode = staticMethod(codec, "decode");
    if (info.words == 0 or info.encode == nullptr or info.decode == nullptr) {
        diags.error(loc, "codec '" + codec->getQualifiedNameAsString()
                             + "' needs 'static constexpr std::uint32_t words', 'static std::array<uint, words> "
                               "encode(Value const &)' and 'static Value decode(std::array<uint, words> const &)'");
        return std::nullopt;
    }
    auto const returned = asStdArray(info.encode->getReturnType());
    if (not returned or returned->size != info.words
        or not bareType(returned->element)->isSpecificBuiltinType(clang::BuiltinType::UInt)) {
        diags.error(info.encode->getLocation(), "encode must return std::array<uint, words>");
        return std::nullopt;
    }
    return info;
}

std::optional<std::uint32_t> packedWords(clang::QualType const original, clang::ASTContext &context,
                                         Diagnostics &diags, clang::SourceLocation const loc) {
    if (asTcSpecialization(original, "Encoded") != nullptr) {
        auto const info = encodedInfo(original, context, diags, loc);
        return info ? std::optional<std::uint32_t>{info->words} : std::nullopt;
    }
    auto type = bareType(original);
    if (auto const *enumType = type->getAs<clang::EnumType>()) {
        type = bareType(enumType->getDecl()->getIntegerType());
    }
    if (type->isBooleanType()) {
        return 1;
    }
    if (type->isSpecificBuiltinType(clang::BuiltinType::Float)) {
        return 1;
    }
    if (type->isSpecificBuiltinType(clang::BuiltinType::Double)) {
        return 2;
    }
    if (type->isIntegerType() and context.getTypeSize(type) == 32) {
        return 1;
    }
    if (auto const vec = asTcVec(type)) {
        auto const element = packedWords(vec->element, context, diags, loc);
        return element ? std::optional<std::uint32_t>{vec->size * *element} : std::nullopt;
    }
    if (auto const matrix = asTcMat(type)) {
        auto const element = packedWords(matrix->element, context, diags, loc);
        return element ? std::optional<std::uint32_t>{matrix->columns * matrix->rows * *element} : std::nullopt;
    }
    if (auto const array = asStdArray(type)) {
        auto const element = packedWords(array->element, context, diags, loc);
        return element ? std::optional<std::uint32_t>{static_cast<std::uint32_t>(array->size) * *element}
                       : std::nullopt;
    }
    if (type->isArrayType()) {
        diags.error(loc, "C arrays cannot be stored in tables; use std::array");
        return std::nullopt;
    }
    if (auto const *record = recordSchema(type)) {
        return recordWords(record, original, context, diags, loc);
    }
    diags.error(loc, "'" + original.getAsString()
                         + "' cannot be stored in a table; use bool, int32_t, uint32_t, float, double, 32-bit "
                           "enums, tc vectors and matrices, std::array or structs of these");
    return std::nullopt;
}

std::optional<TableAddressing> tableAddressing(clang::QualType const layout, Diagnostics &diags,
                                               clang::SourceLocation const loc) {
    auto const *record = bareType(layout)->getAsCXXRecordDecl();
    record = record != nullptr ? record->getDefinition() : nullptr;
    auto addressing = TableAddressing{};
    auto contiguous = std::optional<bool>{};
    if (record != nullptr) {
        addressing.record = record;
        addressing.fieldAddress = staticMethod(record, "fieldAddress");
        addressing.fieldAlignment = staticMethod(record, "fieldAlignment");
        for (auto const *decl : record->decls()) {
            auto const *var = llvm::dyn_cast<clang::VarDecl>(decl);
            // Static members of class templates have an initializer only once used.
            auto const *value = var != nullptr and var->getName() == "contiguousRecords" and var->isConstexpr()
                                        and var->getInit() != nullptr
                                  ? var->evaluateValue()
                                  : nullptr;
            if (value != nullptr and value->isInt()) {
                contiguous = value->getInt().getBoolValue();
            }
        }
    }
    auto const hasBodies = addressing.fieldAddress != nullptr and addressing.fieldAddress->hasBody()
                       and addressing.fieldAlignment != nullptr and addressing.fieldAlignment->hasBody();
    if (not contiguous or not hasBodies) {
        diags.error(loc, "table layout '" + layout.getAsString()
                             + "' needs 'static constexpr bool contiguousRecords' and constexpr static member "
                               "functions fieldAddress and fieldAlignment with bodies (see tc/table.hpp)");
        return std::nullopt;
    }
    addressing.contiguousRecords = *contiguous;
    return addressing;
}

std::optional<std::uint64_t> evaluateCall(clang::ASTContext &context, clang::FunctionDecl const *function,
                                          std::vector<std::uint64_t> const &arguments) {
    if (function->getNumParams() != arguments.size()) {
        return std::nullopt;
    }
    auto *callee = const_cast<clang::FunctionDecl *>(function);
    auto const loc = function->getLocation();
    auto *reference = clang::DeclRefExpr::Create(context, clang::NestedNameSpecifierLoc{}, clang::SourceLocation{},
                                                 callee, false, loc, callee->getType(), clang::VK_LValue);
    auto *pointer =
        clang::ImplicitCastExpr::Create(context, context.getPointerType(callee->getType()),
                                        clang::CK_FunctionToPointerDecay, reference, nullptr, clang::VK_PRValue,
                                        clang::FPOptionsOverride{});
    auto values = std::vector<clang::Expr *>{};
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        auto const type = callee->getParamDecl(static_cast<unsigned>(i))->getType().getCanonicalType().getUnqualifiedType();
        if (not type->isIntegerType()) {
            return std::nullopt;
        }
        values.push_back(clang::IntegerLiteral::Create(
            context, llvm::APInt(static_cast<unsigned>(context.getIntWidth(type)), arguments[i]), type, loc));
    }
    auto *call = clang::CallExpr::Create(context, pointer, values, callee->getReturnType().getNonReferenceType(),
                                         clang::VK_PRValue, loc, clang::FPOptionsOverride{});
    auto result = clang::Expr::EvalResult{};
    if (not call->EvaluateAsRValue(result, context) or not result.Val.isInt()) {
        return std::nullopt;
    }
    return result.Val.getInt().getZExtValue();
}

std::optional<PackedLayout> packedLayout(clang::QualType const element, clang::ASTContext &context,
                                         Diagnostics &diags, clang::SourceLocation const loc) {
    auto const type = bareType(element);
    if (not type->isRecordType()) {
        diags.error(loc, "table elements are structs, vectors or matrices; store '" + element.getAsString()
                             + "' in a BufferBinding, which is already compact");
        return std::nullopt;
    }
    auto layout = PackedLayout{};
    auto const *record = recordSchema(type);
    if (record == nullptr) {
        auto const words = packedWords(type, context, diags, loc);
        if (not words) {
            return std::nullopt;
        }
        layout.words = *words;
        layout.fields.push_back(PackedField{nullptr, type, 0, *words});
        return layout;
    }
    auto const total = recordWords(record, element, context, diags, loc);
    if (not total) {
        return std::nullopt;
    }
    layout.words = *total;
    layout.isRecord = true;
    auto offset = std::uint32_t{0};
    for (auto const *field : record->fields()) {
        auto const words = *packedWords(field->getType(), context, diags, field->getLocation());
        layout.fields.push_back(PackedField{field, field->getType(), offset, words});
        offset += words;
    }
    return layout;
}

} // namespace tc::transpiler
