#include "glsl_writer.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <clang/AST/APValue.h>
#include <clang/AST/Expr.h>
#include <clang/AST/ExprCXX.h>
#include <clang/AST/RecordLayout.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/StmtCXX.h>
#include <clang/Basic/OperatorKinds.h>
#include <clang/Basic/SourceManager.h>

#include "glsl_names.hpp"
#include "packed_layout.hpp"
#include "tc_types.hpp"

namespace tc::transpiler {

namespace {

// How the expression being written is used: a buffer element on the
// left of `=` is written, on the left of `+=` read and written.
enum class Role : std::uint8_t {
    Read,
    Write,
    ReadWrite,
};

enum class FunctionKind : std::uint8_t {
    KernelEntry,  // the kernel's main()
    KernelMethod, // other kernel member functions; members become globals
    StructMethod, // non-static member of a user struct; `this` becomes `self`
    Free,         // free functions and static member functions
};

struct FunctionEntry {
    clang::FunctionDecl const *decl = nullptr;
    FunctionKind kind = FunctionKind::Free;
    std::string glslName;
    std::string definition;
};

std::string join(std::vector<std::string> const &parts, char const *separator) {
    auto text = std::string{};
    for (std::size_t i = 0; i < parts.size(); ++i) {
        text += i == 0 ? "" : separator;
        text += parts[i];
    }
    return text;
}

std::string indent(int const depth) {
    return std::string(static_cast<std::size_t>(depth) * 4, ' ');
}

// Shortest literal that reads back as the same float (or double).
std::string formatFloat(double const value, bool const isDouble) {
    auto const roundTrips = [&](char const *text) {
        return isDouble ? std::strtod(text, nullptr) == value
                        : std::strtof(text, nullptr) == static_cast<float>(value);
    };
    char buffer[64] = {};
    for (int precision = 1; precision <= 17; ++precision) {
        std::snprintf(buffer, sizeof buffer, "%.*g", precision, value);
        if (roundTrips(buffer)) {
            break;
        }
    }
    auto const magnitude = std::fabs(value);
    if (std::strchr(buffer, 'e') != nullptr and magnitude >= 1e-4 and magnitude < 1e7) {
        char fixed[64] = {};
        for (int decimals = 1; decimals <= 9; ++decimals) {
            std::snprintf(fixed, sizeof fixed, "%.*f", decimals, value);
            if (roundTrips(fixed)) {
                std::memcpy(buffer, fixed, sizeof buffer);
                break;
            }
        }
    }
    auto text = std::string{buffer};
    if (text.find_first_of(".e") == std::string::npos) {
        text += ".0";
    }
    return isDouble ? text + "lf" : text;
}

bool isNonConstLValueRef(clang::QualType const type) {
    return type->isLValueReferenceType() and not type.getNonReferenceType().isConstQualified();
}

bool isLiteralLike(clang::Expr const *e) {
    e = e->IgnoreParens();
    if (auto const *unary = llvm::dyn_cast<clang::UnaryOperator>(e)) {
        if (unary->getOpcode() == clang::UO_Minus or unary->getOpcode() == clang::UO_Plus) {
            return isLiteralLike(unary->getSubExpr());
        }
        return false;
    }
    return llvm::isa<clang::IntegerLiteral, clang::FloatingLiteral, clang::CXXBoolLiteralExpr>(e);
}

bool isScalarConversion(clang::CastKind const kind) {
    switch (kind) {
    case clang::CK_IntegralCast:
    case clang::CK_IntegralToFloating:
    case clang::CK_FloatingToIntegral:
    case clang::CK_FloatingCast:
    case clang::CK_IntegralToBoolean:
    case clang::CK_FloatingToBoolean:
    case clang::CK_BooleanToSignedIntegral:
        return true;
    default:
        return false;
    }
}

bool isTransparentCast(clang::CastKind const kind) {
    switch (kind) {
    case clang::CK_LValueToRValue:
    case clang::CK_NoOp:
    case clang::CK_ArrayToPointerDecay:
    case clang::CK_FunctionToPointerDecay:
    case clang::CK_DerivedToBase:
    case clang::CK_UncheckedDerivedToBase:
    case clang::CK_ConstructorConversion:
    case clang::CK_UserDefinedConversion:
    case clang::CK_ToVoid:
        return true;
    default:
        return false;
    }
}

struct Writer {
    Writer(KernelModel &kernel_, clang::ASTContext &context_, Diagnostics &diags_)
        : kernel(kernel_), context(context_), diags(diags_) {}

    std::optional<GlslBody> run();

private:
    // Types.
    std::string typeName(clang::QualType type, clang::SourceLocation loc);
    std::string declaration(clang::QualType type, std::string const &name, clang::SourceLocation loc);
    std::string recordName(clang::CXXRecordDecl const *record, clang::SourceLocation loc);
    std::string requireStruct(clang::CXXRecordDecl const *record, clang::SourceLocation loc);
    std::string fieldName(clang::FieldDecl const *field) { return glslIdentifier(field->getName()); }
    std::string zeroValue(clang::QualType type, clang::SourceLocation loc);

    // Constants.
    std::string constantValue(clang::APValue const &value, clang::QualType type, clang::SourceLocation loc);
    std::string arrayConstant(clang::APValue const &value, clang::QualType element, std::uint64_t size,
                              clang::SourceLocation loc);
    std::string requireKernelConstant(clang::FieldDecl const *field);
    std::string requireGlobalConstant(clang::VarDecl const *var, clang::SourceLocation useLoc);
    void defineConstant(std::string const &name, clang::APValue const &value, clang::QualType type,
                        clang::SourceLocation loc);
    void elementInitializers(std::string const &target, clang::APValue const &value, clang::QualType type,
                             clang::SourceLocation loc);
    bool containsVec3Struct(clang::QualType type);
    std::string uniqueGlobalName(std::string const &base);

    // Functions.
    std::string requireFunction(clang::FunctionDecl const *decl, FunctionKind kind, clang::SourceLocation loc);
    std::string functionBaseName(clang::FunctionDecl const *decl, clang::SourceLocation loc);
    std::string signature(FunctionEntry const &entry);
    std::string definition(FunctionEntry const &entry);
    bool checkRecursion();

    // Statements.
    std::string stmt(clang::Stmt const *s, int depth);
    std::string block(clang::Stmt const *s, int depth);
    std::string ifStatement(clang::IfStmt const *s, int depth);
    std::string varDecl(clang::VarDecl const *var);
    std::string forInit(clang::Stmt const *init);
    std::string forRange(clang::CXXForRangeStmt const *s, int depth);
    std::string localName(clang::ValueDecl const *decl);

    // Expressions.
    std::string expr(clang::Expr const *e);
    std::string exprAs(clang::Expr const *e, Role role);
    std::string implicitCast(clang::ImplicitCastExpr const *cast, Role inherited);
    std::string explicitCast(clang::ExplicitCastExpr const *cast);
    std::string scalarConversion(clang::CastExpr const *cast);
    std::string binary(clang::BinaryOperator const *op);
    std::string unary(clang::UnaryOperator const *op, Role inherited);
    std::string declRef(clang::DeclRefExpr const *ref, Role inherited);
    std::string member(clang::MemberExpr const *me, Role inherited);
    std::string kernelMember(clang::FieldDecl const *field, Role inherited, clang::SourceLocation loc);
    std::string memberCall(clang::CXXMemberCallExpr const *call, Role inherited);
    std::string operatorCall(clang::CXXOperatorCallExpr const *call, Role inherited);
    std::string call(clang::CallExpr const *call);
    std::string arguments(clang::FunctionDecl const *callee, llvm::ArrayRef<clang::Expr const *> args);
    std::string construct(clang::CXXConstructExpr const *ctor, Role inherited);
    std::string initList(clang::InitListExpr const *list);
    std::string arrayInit(clang::InitListExpr const *list, clang::QualType element, std::uint64_t size);
    std::string structValue(clang::CXXRecordDecl const *record, clang::SourceLocation loc);

    // Tables: `t[i]` and `t[i].field` reads become calls of generated
    // accessors; writes are rewritten statement by statement.
    struct TableAccess {
        TableModel *table = nullptr;
        clang::Expr const *index = nullptr;
        PackedField const *field = nullptr; // nullptr: the whole element
        clang::Expr const *node = nullptr;  // the subscript, or the member expression selecting `field`
    };
    TableModel *tableOfSubscript(clang::CXXOperatorCallExpr const *call);
    std::optional<TableAccess> tableFieldAccess(clang::MemberExpr const *member);
    std::optional<TableAccess> tableTarget(clang::Expr const *e);
    std::string tableRead(TableAccess const &access, Role inherited, clang::SourceLocation loc);
    std::optional<std::string> tableWriteStatement(clang::Expr const *e, int depth);
    std::string tableIndex(clang::Expr const *index);
    // The words an accessor of element `i` touches: [first, last) counted
    // from a base that is aligned to `vector` words, so they can be moved in
    // uvec2 / uvec4 chunks that never leave the element.
    struct TableWindow {
        std::string chunkBase; // index of chunk 0 in the `vector`-word view
        std::uint32_t vector = 1;
        std::uint32_t first = 0;
        std::uint32_t last = 0;
    };
    TableWindow tableWindow(TableModel &table, PackedField const *field);
    std::string tableView(TableModel &table, std::uint32_t vector);
    std::string tableGetter(TableModel &table, PackedField const *field);
    std::string tableSetter(TableModel &table, PackedField const *field);
    std::string unpackValue(clang::QualType type, std::vector<std::string> const &words, std::uint32_t &cursor,
                            clang::SourceLocation loc);
    void packWords(clang::QualType type, std::string const &value, std::vector<std::string> &words,
                   clang::SourceLocation loc);

    bool isAtomicTarget(clang::Expr const *e) const;
    void markBufferUse(clang::Expr const *object, Role role);
    void markImageUse(clang::Expr const *image, bool write);
    clang::FieldDecl const *kernelField(clang::Expr const *e) const;
    bool inKernelScope() const {
        return currentKind == FunctionKind::KernelEntry or currentKind == FunctionKind::KernelMethod;
    }
    void unsupported(clang::SourceLocation loc, std::string const &what) {
        diags.error(loc, what + " is not supported in GPU kernels");
    }

    KernelModel &kernel;
    clang::ASTContext &context;
    Diagnostics &diags;
    GlslBody out;

    std::map<clang::CXXRecordDecl const *, std::string> structNames;
    std::set<clang::CXXRecordDecl const *> structsInProgress;
    std::map<clang::Decl const *, std::string> constantNames;
    std::set<std::string> globalNames;
    std::map<clang::FunctionDecl const *, std::size_t> functionIndex;
    std::vector<FunctionEntry> functions;
    std::deque<std::size_t> pending;
    std::map<clang::FunctionDecl const *, std::set<clang::FunctionDecl const *>> callGraph;
    std::map<clang::ValueDecl const *, std::string> localNames;
    std::vector<std::string> constantInitializers;
    std::set<std::string> definedAccessors;
    // While rewriting a table write, the access that a local copy replaces.
    struct Replacement {
        clang::Expr const *node = nullptr;
        std::string name;
    } replacement;

    clang::FunctionDecl const *currentFunction = nullptr;
    FunctionKind currentKind = FunctionKind::KernelEntry;
    Role role = Role::Read;
    int loopCounter = 0;
};

// ---------------------------------------------------------------- types

std::string Writer::typeName(clang::QualType const original, clang::SourceLocation const loc) {
    auto const type = bareType(original);
    if (type->isVoidType()) {
        return "void";
    }
    if (auto const *enumType = type->getAs<clang::EnumType>()) {
        return typeName(enumType->getDecl()->getIntegerType(), loc);
    }
    if (auto const *builtin = type->getAs<clang::BuiltinType>()) {
        switch (builtin->getKind()) {
        case clang::BuiltinType::Bool:
            return "bool";
        case clang::BuiltinType::Int:
        case clang::BuiltinType::Long:
        case clang::BuiltinType::LongLong:
            return "int";
        case clang::BuiltinType::UInt:
        case clang::BuiltinType::ULong:
        case clang::BuiltinType::ULongLong:
            return "uint";
        case clang::BuiltinType::Float:
            return "float";
        case clang::BuiltinType::Double:
            out.usesDouble = true;
            return "double";
        default:
            diags.error(loc, "type '" + original.getAsString()
                                 + "' has no GLSL equivalent; use int, uint, float, double or bool");
            return "int";
        }
    }
    if (auto const vec = asTcVec(type)) {
        auto const element = typeName(vec->element, loc);
        auto const prefix = element == "float" ? "" : element == "double" ? "d" : element == "int" ? "i"
                          : element == "uint"  ? "u"
                                               : "b";
        return prefix + std::string{"vec"} + std::to_string(vec->size);
    }
    if (auto const matrix = asTcMat(type)) {
        auto const prefix = typeName(matrix->element, loc) == "double" ? "dmat" : "mat";
        auto const shape = matrix->columns == matrix->rows
                             ? std::to_string(matrix->columns)
                             : std::to_string(matrix->columns) + "x" + std::to_string(matrix->rows);
        return prefix + shape;
    }
    if (auto const array = asStdArray(type)) {
        return typeName(array->element, loc) + "[" + std::to_string(array->size) + "]";
    }
    if (auto const *array = context.getAsConstantArrayType(type)) {
        return typeName(array->getElementType(), loc) + "[" + std::to_string(array->getSize().getZExtValue()) + "]";
    }
    if (type->isPointerType() or type->isArrayType()) {
        unsupported(loc, "pointer or unsized array type '" + original.getAsString() + "'");
        return "int";
    }
    if (auto const *encoded = asTcSpecialization(type, "Encoded")) {
        return typeName(templateTypeArg(encoded, 0), loc); // only tables store the encoding
    }
    if (auto const *record = type->getAsCXXRecordDecl()) {
        if (isTcDecl(record)) {
            diags.error(loc, "'" + original.getAsString() + "' cannot be used as a value inside a kernel");
            return "int";
        }
        return requireStruct(record, loc);
    }
    diags.error(loc, "type '" + original.getAsString() + "' has no GLSL equivalent");
    return "int";
}

std::string Writer::declaration(clang::QualType const original, std::string const &name,
                                clang::SourceLocation const loc) {
    auto type = bareType(original);
    auto dims = std::string{};
    for (;;) {
        if (auto const array = asStdArray(type)) {
            dims += "[" + std::to_string(array->size) + "]";
            type = bareType(array->element);
        } else if (auto const *carray = context.getAsConstantArrayType(type)) {
            dims += "[" + std::to_string(carray->getSize().getZExtValue()) + "]";
            type = bareType(carray->getElementType());
        } else {
            break;
        }
    }
    return typeName(type, loc) + " " + name + dims;
}

// The GLSL spelling of a record name: template arguments are appended, and
// tc records (codecs) get a tc_ prefix so they cannot clash with user names.
std::string Writer::recordName(clang::CXXRecordDecl const *record, clang::SourceLocation const loc) {
    auto name = (isTcDecl(record) ? "tc_" : "") + glslIdentifier(record->getName());
    if (auto const *spec = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(record)) {
        for (auto const &arg : spec->getTemplateArgs().asArray()) {
            if (arg.getKind() == clang::TemplateArgument::Type) {
                auto part = typeName(arg.getAsType(), loc);
                for (auto &c : part) {
                    c = (c == '[' or c == ']') ? '_' : c;
                }
                name += "_" + part;
            } else if (arg.getKind() == clang::TemplateArgument::Integral) {
                name += "_" + std::to_string(arg.getAsIntegral().getExtValue());
            }
        }
    }
    return name;
}

std::string Writer::requireStruct(clang::CXXRecordDecl const *record, clang::SourceLocation const loc) {
    record = record->getDefinition();
    if (record == nullptr) {
        diags.error(loc, "struct has no definition");
        return "int";
    }
    if (auto const it = structNames.find(record); it != structNames.end()) {
        return it->second;
    }
    if (structsInProgress.count(record) != 0) {
        diags.error(record->getLocation(), "recursive struct types are not supported");
        return "int";
    }
    if (record->isUnion() or record->isPolymorphic() or record->getNumBases() != 0 or not record->isAggregate()
        or record->getName().empty()) {
        diags.error(record->getLocation(), "only named aggregate structs without base classes or virtual "
                                           "functions can be used in kernels");
        return "int";
    }
    structsInProgress.insert(record);
    auto const name = recordName(record, loc);
    auto text = "struct " + name + " {\n";
    for (auto const *field : record->fields()) {
        if (field->isBitField()) {
            unsupported(field->getLocation(), "bit-field");
        }
        text += "    " + declaration(field->getType(), fieldName(field), field->getLocation()) + ";\n";
    }
    text += "};";
    structsInProgress.erase(record);
    structNames.emplace(record, name);
    out.structs.push_back(text);
    return name;
}

std::string Writer::zeroValue(clang::QualType const original, clang::SourceLocation const loc) {
    auto const type = bareType(original);
    if (type->isBooleanType()) {
        return "false";
    }
    if (type->isIntegerType() or type->isEnumeralType()) {
        return typeName(type, loc) == "uint" ? "0u" : "0";
    }
    if (type->isRealFloatingType()) {
        return typeName(type, loc) == "double" ? "0.0lf" : "0.0";
    }
    if (asTcVec(type)) {
        return typeName(type, loc) + "(0)";
    }
    if (asTcMat(type)) {
        return typeName(type, loc) + "(0.0)";
    }
    if (auto const array = asStdArray(type)) {
        auto elements = std::vector<std::string>(array->size, zeroValue(array->element, loc));
        return typeName(type, loc) + "(" + join(elements, ", ") + ")";
    }
    if (auto const *carray = context.getAsConstantArrayType(type)) {
        auto elements =
            std::vector<std::string>(carray->getSize().getZExtValue(), zeroValue(carray->getElementType(), loc));
        return typeName(type, loc) + "(" + join(elements, ", ") + ")";
    }
    if (auto const *encoded = asTcSpecialization(type, "Encoded")) {
        return zeroValue(templateTypeArg(encoded, 0), loc);
    }
    if (auto const *record = type->getAsCXXRecordDecl()) {
        return structValue(record, loc);
    }
    diags.error(loc, "cannot build a zero value of '" + original.getAsString() + "'");
    return "0";
}

// A struct built from its default member initializers (or zeros).
std::string Writer::structValue(clang::CXXRecordDecl const *record, clang::SourceLocation const loc) {
    auto const name = requireStruct(record, loc);
    auto fields = std::vector<std::string>{};
    for (auto const *field : record->fields()) {
        fields.push_back(field->hasInClassInitializer() ? expr(field->getInClassInitializer())
                                                        : zeroValue(field->getType(), loc));
    }
    return name + "(" + join(fields, ", ") + ")";
}

// ------------------------------------------------------------- constants

std::string Writer::constantValue(clang::APValue const &value, clang::QualType const original,
                                  clang::SourceLocation const loc) {
    auto const type = bareType(original);
    if (type->isBooleanType()) {
        return value.getInt().getBoolValue() ? "true" : "false";
    }
    if (type->isIntegerType() or type->isEnumeralType()) {
        auto const name = typeName(type, loc);
        auto const &integer = value.getInt();
        return name == "uint" ? std::to_string(integer.getZExtValue()) + "u" : std::to_string(integer.getSExtValue());
    }
    if (type->isRealFloatingType()) {
        auto const isDouble = typeName(type, loc) == "double";
        auto const number = isDouble ? value.getFloat().convertToDouble()
                                     : static_cast<double>(value.getFloat().convertToFloat());
        if (not std::isfinite(number)) {
            diags.error(loc, "infinite or NaN constants cannot be written as GLSL literals");
            return "0.0";
        }
        return formatFloat(number, isDouble);
    }
    if (auto const vec = asTcVec(type)) {
        auto const &fields = value.getStructBase(0);
        auto parts = std::vector<std::string>{};
        for (unsigned i = 0; i < vec->size; ++i) {
            parts.push_back(constantValue(fields.getStructField(i), vec->element, loc));
        }
        return typeName(type, loc) + "(" + join(parts, ", ") + ")";
    }
    if (auto const matrix = asTcMat(type)) {
        auto const &columns = value.getStructField(0);
        auto parts = std::vector<std::string>{};
        for (unsigned c = 0; c < matrix->columns; ++c) {
            auto const &item = c < columns.getArrayInitializedElts() ? columns.getArrayInitializedElt(c)
                                                                      : columns.getArrayFiller();
            auto const &column = item.getStructBase(0);
            for (unsigned r = 0; r < matrix->rows; ++r) {
                parts.push_back(constantValue(column.getStructField(r), matrix->element, loc));
            }
        }
        return typeName(type, loc) + "(" + join(parts, ", ") + ")";
    }
    if (auto const array = asStdArray(type)) {
        auto const element = array->element;
        return arrayConstant(value.getStructField(0), element, array->size, loc);
    }
    if (auto const *carray = context.getAsConstantArrayType(type)) {
        return arrayConstant(value, carray->getElementType(), carray->getSize().getZExtValue(), loc);
    }
    if (auto const *encoded = asTcSpecialization(type, "Encoded")) {
        return constantValue(value.getStructField(0), templateTypeArg(encoded, 0), loc);
    }
    if (auto const *record = type->getAsCXXRecordDecl()) {
        auto const name = requireStruct(record, loc);
        auto parts = std::vector<std::string>{};
        auto index = 0u;
        for (auto const *field : record->fields()) {
            parts.push_back(constantValue(value.getStructField(index++), field->getType(), loc));
        }
        return name + "(" + join(parts, ", ") + ")";
    }
    diags.error(loc, "cannot write a constant of type '" + original.getAsString() + "'");
    return "0";
}

std::string Writer::arrayConstant(clang::APValue const &value, clang::QualType const element,
                                  std::uint64_t const size, clang::SourceLocation const loc) {
    auto parts = std::vector<std::string>{};
    for (std::uint64_t i = 0; i < size; ++i) {
        auto const &item = i < value.getArrayInitializedElts() ? value.getArrayInitializedElt(static_cast<unsigned>(i))
                                                               : value.getArrayFiller();
        parts.push_back(constantValue(item, element, loc));
    }
    return typeName(element, loc) + "[" + std::to_string(size) + "](" + join(parts, ", ") + ")";
}

std::string Writer::uniqueGlobalName(std::string const &base) {
    auto name = base;
    for (int suffix = 1; globalNames.count(name) != 0; ++suffix) {
        name = base + "_" + std::to_string(suffix);
    }
    globalNames.insert(name);
    return name;
}

std::string Writer::requireKernelConstant(clang::FieldDecl const *field) {
    if (auto const it = constantNames.find(field); it != constantNames.end()) {
        return it->second;
    }
    auto const name = uniqueGlobalName(fieldName(field));
    constantNames.emplace(field, name);
    auto const *init = field->getInClassInitializer();
    if (init == nullptr) {
        diags.error(field->getLocation(), "kernel member '" + field->getName().str()
                                              + "' needs an initializer: on the GPU it is a compile-time constant "
                                                "(use tc::Uniform for values set by the host)");
        return name;
    }
    auto result = clang::Expr::EvalResult{};
    if (not init->EvaluateAsRValue(result, context, true)) {
        diags.error(init->getBeginLoc(), "initializer of kernel member '" + field->getName().str()
                                             + "' must be a constant expression");
        return name;
    }
    if (not field->getType().isConstQualified() and field->getName() != "local_size") {
        diags.warning(field->getLocation(), "'" + field->getName().str()
                                                + "' becomes a GLSL constant, so host-side changes never reach the "
                                                  "GPU; declare it 'static constexpr'");
    }
    defineConstant(name, result.Val, field->getType(), field->getLocation());
    return name;
}

bool Writer::containsVec3Struct(clang::QualType const original) {
    auto const type = bareType(original);
    if (auto const array = asStdArray(type)) {
        return containsVec3Struct(array->element);
    }
    if (auto const *carray = context.getAsConstantArrayType(type)) {
        return containsVec3Struct(carray->getElementType());
    }
    auto const *record = type->getAsCXXRecordDecl();
    if (record == nullptr or isTcDecl(record)) {
        return false;
    }
    for (auto const *field : record->fields()) {
        auto const vec = asTcVec(field->getType());
        auto const matrix = asTcMat(field->getType());
        if ((vec and vec->size == 3) or (matrix and matrix->rows == 3) or containsVec3Struct(field->getType())) {
            return true;
        }
    }
    return false;
}

// NVIDIA's Vulkan driver (595.x) misreads constant arrays of structs that
// contain 3-component vectors when glslang copies them into function memory
// for dynamic indexing. Such arrays become globals filled element by element
// at the start of main(), which every driver handles correctly.
void Writer::defineConstant(std::string const &name, clang::APValue const &value, clang::QualType const type,
                            clang::SourceLocation const loc) {
    auto const isArray = asStdArray(type).has_value() or context.getAsConstantArrayType(bareType(type)) != nullptr;
    if (isArray and containsVec3Struct(type)) {
        out.constants.push_back(declaration(type, name, loc) + ";");
        elementInitializers(name, value, type, loc);
        return;
    }
    out.constants.push_back("const " + declaration(type, name, loc) + " = " + constantValue(value, type, loc) + ";");
}

void Writer::elementInitializers(std::string const &target, clang::APValue const &value,
                                 clang::QualType const original, clang::SourceLocation const loc) {
    auto const type = bareType(original);
    auto element = clang::QualType{};
    auto size = std::uint64_t{0};
    auto const *items = &value;
    if (auto const array = asStdArray(type)) {
        element = array->element;
        size = array->size;
        items = &value.getStructField(0);
    } else if (auto const *carray = context.getAsConstantArrayType(type)) {
        element = carray->getElementType();
        size = carray->getSize().getZExtValue();
    } else {
        constantInitializers.push_back(target + " = " + constantValue(value, type, loc) + ";");
        return;
    }
    for (std::uint64_t i = 0; i < size; ++i) {
        auto const &item = i < items->getArrayInitializedElts()
                             ? items->getArrayInitializedElt(static_cast<unsigned>(i))
                             : items->getArrayFiller();
        elementInitializers(target + "[" + std::to_string(i) + "]", item, element, loc);
    }
}

std::string Writer::requireGlobalConstant(clang::VarDecl const *var, clang::SourceLocation const useLoc) {
    if (auto const it = constantNames.find(var); it != constantNames.end()) {
        return it->second;
    }
    auto const name = uniqueGlobalName(glslIdentifier(var->getName()));
    constantNames.emplace(var, name);
    auto const *value = var->evaluateValue();
    if (not var->isConstexpr() or value == nullptr) {
        diags.error(useLoc, "global variable '" + var->getName().str()
                                + "' must be constexpr to be used in a kernel");
        return name;
    }
    defineConstant(name, *value, var->getType(), var->getLocation());
    return name;
}

// ------------------------------------------------------------- functions

std::string Writer::requireFunction(clang::FunctionDecl const *decl, FunctionKind const kind,
                                    clang::SourceLocation const loc) {
    if (auto const *definition = decl->getDefinition()) {
        decl = definition;
    }
    if (currentFunction != nullptr) {
        callGraph[currentFunction].insert(decl);
    }
    if (auto const it = functionIndex.find(decl); it != functionIndex.end()) {
        return functions[it->second].glslName;
    }
    if (not decl->hasBody()) {
        diags.error(loc, "function '" + decl->getNameAsString() + "' has no visible definition");
        return "missing";
    }
    if (decl->isVariadic()) {
        unsupported(decl->getLocation(), "variadic function");
    }
    if (auto const *method = llvm::dyn_cast<clang::CXXMethodDecl>(decl); method != nullptr and method->isVirtual()) {
        unsupported(decl->getLocation(), "virtual function");
    }
    if (decl->getReturnType()->isReferenceType() or decl->getReturnType()->isPointerType()) {
        unsupported(decl->getLocation(), "returning a reference or pointer");
        if (decl->isOverloadedOperator()) {
            diags.note(decl->getLocation(), "operators such as += that return *this cannot be translated; return the "
                                            "result by value (`T operator+(T, T)`) and assign it");
        }
    }
    auto entry = FunctionEntry{};
    entry.decl = decl;
    entry.kind = kind;
    auto const baseName = functionBaseName(decl, loc);
    switch (kind) {
    case FunctionKind::KernelEntry:
        entry.glslName = "main";
        break;
    case FunctionKind::StructMethod:
        entry.glslName = requireStruct(llvm::cast<clang::CXXMethodDecl>(decl)->getParent(), loc) + "_" + baseName;
        break;
    case FunctionKind::Free:
        if (auto const *method = llvm::dyn_cast<clang::CXXMethodDecl>(decl);
            method != nullptr and method->getParent() != kernel.record) {
            // A static member function: no struct value is involved.
            entry.glslName = recordName(method->getParent(), loc) + "_" + baseName;
        } else {
            entry.glslName = baseName;
        }
        break;
    case FunctionKind::KernelMethod:
        entry.glslName = baseName;
        break;
    }
    // GLSL overloads only on parameter types; template instances such as
    // scale<2> and scale<3> would collide, as would a function and a constant.
    if (kind != FunctionKind::KernelEntry) {
        entry.glslName = uniqueGlobalName(entry.glslName);
    }
    functionIndex.emplace(decl, functions.size());
    pending.push_back(functions.size());
    functions.push_back(entry);
    return entry.glslName;
}

namespace {

char const *operatorWord(clang::OverloadedOperatorKind const op, unsigned const params) {
    switch (op) {
    case clang::OO_Plus:
        return params == 1 ? "pos" : "add";
    case clang::OO_Minus:
        return params == 1 ? "neg" : "sub";
    case clang::OO_Star:
        return "mul";
    case clang::OO_Slash:
        return "div";
    case clang::OO_Percent:
        return "mod";
    case clang::OO_Caret:
        return "xor";
    case clang::OO_Amp:
        return "and";
    case clang::OO_Pipe:
        return "or";
    case clang::OO_Tilde:
        return "compl";
    case clang::OO_Exclaim:
        return "not";
    case clang::OO_Less:
        return "lt";
    case clang::OO_Greater:
        return "gt";
    case clang::OO_LessEqual:
        return "le";
    case clang::OO_GreaterEqual:
        return "ge";
    case clang::OO_EqualEqual:
        return "eq";
    case clang::OO_ExclaimEqual:
        return "ne";
    case clang::OO_LessLess:
        return "shl";
    case clang::OO_GreaterGreater:
        return "shr";
    case clang::OO_AmpAmp:
        return "land";
    case clang::OO_PipePipe:
        return "lor";
    case clang::OO_Subscript:
        return "at";
    case clang::OO_Call:
        return "call";
    default:
        return "op";
    }
}

} // namespace

// Identifier part of a function's GLSL name: operators and conversions get
// words, template instances their arguments (`lerp_float`, `scale_2`).
std::string Writer::functionBaseName(clang::FunctionDecl const *decl, clang::SourceLocation const loc) {
    auto name = std::string{};
    if (decl->getDeclName().isIdentifier()) {
        name = glslIdentifier(decl->getName());
    } else if (decl->isOverloadedOperator()) {
        auto const *method = llvm::dyn_cast<clang::CXXMethodDecl>(decl);
        auto const params = decl->getNumParams() + (method != nullptr and not method->isStatic() ? 1 : 0);
        name = std::string{"op_"} + operatorWord(decl->getOverloadedOperator(), params);
    } else if (auto const *conversion = llvm::dyn_cast<clang::CXXConversionDecl>(decl)) {
        name = "to_" + typeName(conversion->getConversionType(), loc);
    } else {
        name = "fn";
    }
    if (auto const *args = decl->getTemplateSpecializationArgs()) {
        for (auto const &arg : args->asArray()) {
            if (arg.getKind() == clang::TemplateArgument::Type) {
                auto part = typeName(arg.getAsType(), loc);
                for (auto &c : part) {
                    c = std::isalnum(static_cast<unsigned char>(c)) ? c : '_';
                }
                name += "_" + part;
            } else if (arg.getKind() == clang::TemplateArgument::Integral) {
                auto const value = arg.getAsIntegral().getExtValue();
                name += "_" + (value < 0 ? "m" + std::to_string(-value) : std::to_string(value));
            }
        }
    }
    return name;
}

std::string Writer::signature(FunctionEntry const &entry) {
    auto const *decl = entry.decl;
    auto params = std::vector<std::string>{};
    if (entry.kind == FunctionKind::StructMethod) {
        auto const *method = llvm::cast<clang::CXXMethodDecl>(decl);
        auto const self = requireStruct(method->getParent(), decl->getLocation()) + " self";
        params.push_back(method->isConst() ? self : "inout " + self);
    }
    auto index = 0;
    for (auto const *param : decl->parameters()) {
        auto name = param->getName().empty() ? "tc_arg" + std::to_string(index) : localName(param);
        auto const qualifier = isNonConstLValueRef(param->getType()) ? "inout " : "";
        if (param->getType()->isPointerType()) {
            unsupported(param->getLocation(), "pointer parameter");
        }
        params.push_back(qualifier + declaration(param->getType().getNonReferenceType(), name, param->getLocation()));
        ++index;
    }
    return typeName(decl->getReturnType(), decl->getLocation()) + " " + entry.glslName + "(" + join(params, ", ")
         + ")";
}

std::string Writer::definition(FunctionEntry const &entry) {
    currentFunction = entry.decl;
    currentKind = entry.kind;
    auto const head = signature(entry);
    return head + " " + block(entry.decl->getBody(), 0);
}

bool Writer::checkRecursion() {
    enum class Mark : std::uint8_t { Visiting, Done };
    auto marks = std::map<clang::FunctionDecl const *, Mark>{};
    auto ok = true;
    auto const visit = [&](auto const &self, clang::FunctionDecl const *fn) -> void {
        marks[fn] = Mark::Visiting;
        for (auto const *callee : callGraph[fn]) {
            auto const it = marks.find(callee);
            if (it == marks.end()) {
                self(self, callee);
            } else if (it->second == Mark::Visiting) {
                diags.error(callee->getLocation(), "'" + callee->getNameAsString()
                                                       + "' is called recursively; GLSL does not support recursion");
                ok = false;
            }
        }
        marks[fn] = Mark::Done;
    };
    for (auto const &entry : functions) {
        if (marks.count(entry.decl) == 0) {
            visit(visit, entry.decl);
        }
    }
    return ok;
}

// ------------------------------------------------------------ statements

std::string Writer::localName(clang::ValueDecl const *decl) {
    if (auto const it = localNames.find(decl); it != localNames.end()) {
        return it->second;
    }
    auto const name = glslIdentifier(decl->getName());
    localNames.emplace(decl, name);
    return name;
}

std::string Writer::block(clang::Stmt const *s, int const depth) {
    auto text = std::string{"{\n"};
    if (auto const *compound = llvm::dyn_cast<clang::CompoundStmt>(s)) {
        for (auto const *child : compound->body()) {
            text += stmt(child, depth + 1);
        }
    } else {
        text += stmt(s, depth + 1);
    }
    return text + indent(depth) + "}";
}

std::string Writer::ifStatement(clang::IfStmt const *s, int const depth) {
    if (s->getConditionVariable() != nullptr) {
        unsupported(s->getBeginLoc(), "a variable declared in an if condition");
    }
    auto text = "if (" + expr(s->getCond()) + ") " + block(s->getThen(), depth);
    if (auto const *otherwise = s->getElse()) {
        auto const *elseIf = llvm::dyn_cast<clang::IfStmt>(otherwise);
        if (elseIf != nullptr and elseIf->getInit() == nullptr and not elseIf->isConstexpr()) {
            text += " else " + ifStatement(elseIf, depth);
        } else {
            text += " else " + block(otherwise, depth);
        }
    }
    return text;
}

std::string Writer::stmt(clang::Stmt const *s, int const depth) {
    auto const pad = indent(depth);
    if (llvm::isa<clang::CompoundStmt>(s)) {
        return pad + block(s, depth) + "\n";
    }
    if (auto const *decls = llvm::dyn_cast<clang::DeclStmt>(s)) {
        auto text = std::string{};
        for (auto const *decl : decls->decls()) {
            if (auto const *var = llvm::dyn_cast<clang::VarDecl>(decl)) {
                text += pad + varDecl(var) + ";\n";
            } else if (not llvm::isa<clang::TypedefNameDecl, clang::UsingDecl, clang::UsingDirectiveDecl,
                                     clang::StaticAssertDecl>(decl)) {
                unsupported(decl->getLocation(), "this local declaration");
            }
        }
        return text;
    }
    if (llvm::isa<clang::NullStmt>(s)) {
        return pad + ";\n";
    }
    if (auto const *ifs = llvm::dyn_cast<clang::IfStmt>(s)) {
        if (ifs->isConstexpr()) {
            auto taken = false;
            if (not ifs->getCond()->EvaluateAsBooleanCondition(taken, context)) {
                diags.error(ifs->getBeginLoc(), "cannot evaluate 'if constexpr' condition");
            }
            auto const *branch = taken ? ifs->getThen() : ifs->getElse();
            return branch != nullptr ? stmt(branch, depth) : std::string{};
        }
        if (auto const *init = ifs->getInit()) {
            return pad + "{\n" + stmt(init, depth + 1) + indent(depth + 1) + ifStatement(ifs, depth + 1) + "\n" + pad
                 + "}\n";
        }
        return pad + ifStatement(ifs, depth) + "\n";
    }
    if (auto const *loop = llvm::dyn_cast<clang::ForStmt>(s)) {
        if (loop->getConditionVariable() != nullptr) {
            unsupported(loop->getBeginLoc(), "a variable declared in a for condition");
        }
        auto const init = loop->getInit() != nullptr ? forInit(loop->getInit()) : std::string{};
        auto const cond = loop->getCond() != nullptr ? expr(loop->getCond()) : std::string{};
        auto const inc = loop->getInc() != nullptr ? expr(loop->getInc()) : std::string{};
        return pad + "for (" + init + "; " + cond + "; " + inc + ") " + block(loop->getBody(), depth) + "\n";
    }
    if (auto const *loop = llvm::dyn_cast<clang::CXXForRangeStmt>(s)) {
        return forRange(loop, depth);
    }
    if (auto const *loop = llvm::dyn_cast<clang::WhileStmt>(s)) {
        if (loop->getConditionVariable() != nullptr) {
            unsupported(loop->getBeginLoc(), "a variable declared in a while condition");
        }
        return pad + "while (" + expr(loop->getCond()) + ") " + block(loop->getBody(), depth) + "\n";
    }
    if (auto const *loop = llvm::dyn_cast<clang::DoStmt>(s)) {
        return pad + "do " + block(loop->getBody(), depth) + " while (" + expr(loop->getCond()) + ");\n";
    }
    if (llvm::isa<clang::BreakStmt>(s)) {
        return pad + "break;\n";
    }
    if (llvm::isa<clang::ContinueStmt>(s)) {
        return pad + "continue;\n";
    }
    if (auto const *ret = llvm::dyn_cast<clang::ReturnStmt>(s)) {
        auto const *value = ret->getRetValue();
        return pad + (value != nullptr ? "return " + expr(value) + ";\n" : std::string{"return;\n"});
    }
    if (auto const *sw = llvm::dyn_cast<clang::SwitchStmt>(s)) {
        if (sw->getInit() != nullptr or sw->getConditionVariable() != nullptr) {
            unsupported(sw->getBeginLoc(), "a switch with an init statement");
        }
        return pad + "switch (" + expr(sw->getCond()) + ") " + block(sw->getBody(), depth) + "\n";
    }
    if (auto const *label = llvm::dyn_cast<clang::CaseStmt>(s)) {
        auto value = clang::Expr::EvalResult{};
        if (label->getRHS() != nullptr or not label->getLHS()->EvaluateAsRValue(value, context)) {
            unsupported(label->getBeginLoc(), "this case label");
            return {};
        }
        auto const cond = constantValue(value.Val, label->getLHS()->getType(), label->getBeginLoc());
        return indent(depth - 1) + "case " + cond + ":\n" + stmt(label->getSubStmt(), depth);
    }
    if (auto const *label = llvm::dyn_cast<clang::DefaultStmt>(s)) {
        return indent(depth - 1) + "default:\n" + stmt(label->getSubStmt(), depth);
    }
    if (auto const *attributed = llvm::dyn_cast<clang::AttributedStmt>(s)) {
        return stmt(attributed->getSubStmt(), depth);
    }
    if (auto const *e = llvm::dyn_cast<clang::Expr>(s)) {
        if (auto rewritten = tableWriteStatement(e, depth)) {
            return *rewritten;
        }
        return pad + expr(e) + ";\n";
    }
    if (llvm::isa<clang::CXXTryStmt>(s)) {
        diags.error(s->getBeginLoc(), "exceptions are not supported in GPU kernels");
        return {};
    }
    unsupported(s->getBeginLoc(), std::string{"statement '"} + s->getStmtClassName() + "'");
    return {};
}

std::string Writer::forInit(clang::Stmt const *init) {
    auto const *decls = llvm::dyn_cast<clang::DeclStmt>(init);
    if (decls == nullptr) {
        return expr(llvm::cast<clang::Expr>(init));
    }
    auto text = std::string{};
    for (auto const *decl : decls->decls()) {
        auto const *var = llvm::cast<clang::VarDecl>(decl);
        if (text.empty()) {
            text = varDecl(var);
        } else {
            text += ", " + localName(var) + (var->hasInit() ? " = " + expr(var->getInit()) : std::string{});
        }
    }
    return text;
}

std::string Writer::varDecl(clang::VarDecl const *var) {
    if (var->isStaticLocal()) {
        unsupported(var->getLocation(), "static local variable");
    }
    auto type = var->getType();
    if (type->isReferenceType()) {
        if (isNonConstLValueRef(type)) {
            diags.error(var->getLocation(), "GLSL has no references; '" + var->getName().str()
                                                + "' must be a copy or a const reference");
        } else if (var->getInit() != nullptr and tableTarget(var->getInit())) {
            diags.error(var->getLocation(), "'" + var->getName().str()
                                                + "' must be a copy of the table element (drop the '&'): on the "
                                                  "CPU a reference would see later writes, on the GPU it cannot");
        }
        type = type.getNonReferenceType();
    }
    if (type->isPointerType()) {
        unsupported(var->getLocation(), "pointer variable");
    }
    auto const *typedefType = type->getAs<clang::TypedefType>();
    auto const isSizeType = typedefType != nullptr and typedefType->getDecl()->getName().contains("size_t");
    if (bareType(type)->isIntegerType() and context.getTypeSize(bareType(type)) == 64 and not isSizeType) {
        diags.warning(var->getLocation(), "64-bit integer '" + var->getName().str() + "' becomes 32-bit on the GPU");
    }
    auto const isConstant = var->isConstexpr() or (type.isConstQualified() and var->getInit() != nullptr
                                                   and var->getInit()->isCXX11ConstantExpr(context));
    auto text = std::string{isConstant ? "const " : ""} + declaration(type, localName(var), var->getLocation());
    if (auto const *init = var->getInit()) {
        text += " = " + expr(init);
    }
    return text;
}

std::string Writer::forRange(clang::CXXForRangeStmt const *s, int const depth) {
    auto const *range = s->getRangeInit();
    auto const rangeType = bareType(range->getType());
    auto size = std::uint64_t{0};
    if (auto const array = asStdArray(rangeType)) {
        size = array->size;
    } else if (auto const *carray = context.getAsConstantArrayType(rangeType)) {
        size = carray->getSize().getZExtValue();
    } else {
        unsupported(s->getBeginLoc(), "range-for over anything but std::array or a C array");
        return {};
    }
    auto const *var = s->getLoopVariable();
    if (isNonConstLValueRef(var->getType())) {
        diags.error(var->getLocation(), "range-for cannot modify elements on the GPU; use an index loop");
    }
    auto const index = "tc_i" + std::to_string(loopCounter++);
    auto const pad = indent(depth);
    auto text = pad + "for (int " + index + " = 0; " + index + " < " + std::to_string(size) + "; ++" + index + ") {\n";
    text += indent(depth + 1) + declaration(var->getType().getNonReferenceType(), localName(var), var->getLocation())
          + " = " + expr(range) + "[" + index + "];\n";
    if (auto const *compound = llvm::dyn_cast<clang::CompoundStmt>(s->getBody())) {
        for (auto const *child : compound->body()) {
            text += stmt(child, depth + 1);
        }
    } else {
        text += stmt(s->getBody(), depth + 1);
    }
    return text + pad + "}\n";
}

// ----------------------------------------------------------- expressions

std::string Writer::exprAs(clang::Expr const *e, Role const role_) {
    role = role_;
    return expr(e);
}

std::string Writer::expr(clang::Expr const *e) {
    auto const inherited = std::exchange(role, Role::Read);
    auto const loc = e->getBeginLoc();
    if (auto const *literal = llvm::dyn_cast<clang::IntegerLiteral>(e)) {
        auto const isUnsigned = typeName(literal->getType(), loc) == "uint";
        auto const &value = literal->getValue();
        return isUnsigned ? std::to_string(value.getZExtValue()) + "u" : std::to_string(value.getSExtValue());
    }
    if (auto const *literal = llvm::dyn_cast<clang::FloatingLiteral>(e)) {
        auto const isDouble = typeName(literal->getType(), loc) == "double";
        return formatFloat(literal->getValueAsApproximateDouble(), isDouble);
    }
    if (auto const *literal = llvm::dyn_cast<clang::CXXBoolLiteralExpr>(e)) {
        return literal->getValue() ? "true" : "false";
    }
    if (auto const *paren = llvm::dyn_cast<clang::ParenExpr>(e)) {
        return "(" + exprAs(paren->getSubExpr(), inherited) + ")";
    }
    if (auto const *cast = llvm::dyn_cast<clang::ImplicitCastExpr>(e)) {
        return implicitCast(cast, inherited);
    }
    if (auto const *cast = llvm::dyn_cast<clang::ExplicitCastExpr>(e)) {
        return explicitCast(cast);
    }
    if (auto const *op = llvm::dyn_cast<clang::BinaryOperator>(e)) {
        return binary(op);
    }
    if (auto const *op = llvm::dyn_cast<clang::UnaryOperator>(e)) {
        return unary(op, inherited);
    }
    if (auto const *cond = llvm::dyn_cast<clang::ConditionalOperator>(e)) {
        return expr(cond->getCond()) + " ? " + expr(cond->getTrueExpr()) + " : " + expr(cond->getFalseExpr());
    }
    if (auto const *ref = llvm::dyn_cast<clang::DeclRefExpr>(e)) {
        return declRef(ref, inherited);
    }
    if (auto const *me = llvm::dyn_cast<clang::MemberExpr>(e)) {
        return member(me, inherited);
    }
    if (auto const *mc = llvm::dyn_cast<clang::CXXMemberCallExpr>(e)) {
        return memberCall(mc, inherited);
    }
    if (auto const *oc = llvm::dyn_cast<clang::CXXOperatorCallExpr>(e)) {
        return operatorCall(oc, inherited);
    }
    if (llvm::isa<clang::UserDefinedLiteral>(e)) {
        unsupported(loc, "user-defined literal");
        return "0";
    }
    if (auto const *c = llvm::dyn_cast<clang::CallExpr>(e)) {
        return call(c);
    }
    if (auto const *ctor = llvm::dyn_cast<clang::CXXConstructExpr>(e)) {
        return construct(ctor, inherited);
    }
    if (auto const *list = llvm::dyn_cast<clang::InitListExpr>(e)) {
        return initList(list);
    }
    if (auto const *subscript = llvm::dyn_cast<clang::ArraySubscriptExpr>(e)) {
        auto const base = exprAs(subscript->getBase(), inherited);
        return base + "[" + expr(subscript->getIdx()) + "]";
    }
    if (auto const *temp = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(e)) {
        return exprAs(temp->getSubExpr(), inherited);
    }
    if (auto const *bind = llvm::dyn_cast<clang::CXXBindTemporaryExpr>(e)) {
        return exprAs(bind->getSubExpr(), inherited);
    }
    if (auto const *full = llvm::dyn_cast<clang::FullExpr>(e)) {
        return exprAs(full->getSubExpr(), inherited);
    }
    if (auto const *subst = llvm::dyn_cast<clang::SubstNonTypeTemplateParmExpr>(e)) {
        return expr(subst->getReplacement());
    }
    if (auto const *arg = llvm::dyn_cast<clang::CXXDefaultArgExpr>(e)) {
        return expr(arg->getExpr());
    }
    if (auto const *init = llvm::dyn_cast<clang::CXXDefaultInitExpr>(e)) {
        return expr(init->getExpr());
    }
    if (llvm::isa<clang::ImplicitValueInitExpr, clang::CXXScalarValueInitExpr>(e)) {
        return zeroValue(e->getType(), loc);
    }
    if (llvm::isa<clang::UnaryExprOrTypeTraitExpr>(e)) {
        auto value = clang::Expr::EvalResult{};
        if (e->EvaluateAsRValue(value, context)) {
            return constantValue(value.Val, e->getType(), loc);
        }
    }
    if (llvm::isa<clang::CXXThisExpr>(e)) {
        if (currentKind == FunctionKind::StructMethod) {
            return "self";
        }
        diags.error(loc, "'this' can only be used to access kernel members");
        return "this";
    }
    if (llvm::isa<clang::CXXNewExpr>(e)) {
        diags.error(loc, "'new' is not allowed in kernel, only value types are allowed");
        return "0";
    }
    if (llvm::isa<clang::CXXDeleteExpr>(e)) {
        diags.error(loc, "'delete' is not allowed in kernel, only value types are allowed");
        return "0";
    }
    if (llvm::isa<clang::CXXThrowExpr>(e)) {
        diags.error(loc, "exceptions are not supported in GPU kernels");
        return "0";
    }
    if (llvm::isa<clang::LambdaExpr>(e)) {
        diags.error(loc, "lambdas are not supported in GPU kernels; use a member function");
        return "0";
    }
    unsupported(loc, std::string{"expression '"} + e->getStmtClassName() + "'");
    return "0";
}

std::string Writer::implicitCast(clang::ImplicitCastExpr const *cast, Role const inherited) {
    auto const kind = cast->getCastKind();
    if (isScalarConversion(kind)) {
        return scalarConversion(cast);
    }
    if (isTransparentCast(kind)) {
        auto const passRole = kind == clang::CK_LValueToRValue ? Role::Read : inherited;
        return exprAs(cast->getSubExpr(), passRole);
    }
    unsupported(cast->getBeginLoc(), std::string{"implicit conversion '"} + cast->getCastKindName() + "'");
    return "0";
}

std::string Writer::explicitCast(clang::ExplicitCastExpr const *cast) {
    auto const kind = cast->getCastKind();
    if (llvm::isa<clang::CXXConstCastExpr, clang::CXXReinterpretCastExpr, clang::CXXDynamicCastExpr>(cast)) {
        unsupported(cast->getBeginLoc(), "const_cast, reinterpret_cast or dynamic_cast");
        return "0";
    }
    if (isScalarConversion(kind)) {
        return scalarConversion(cast);
    }
    if (isTransparentCast(kind)) {
        return expr(cast->getSubExpr());
    }
    unsupported(cast->getBeginLoc(), std::string{"conversion '"} + cast->getCastKindName() + "'");
    return "0";
}

std::string Writer::scalarConversion(clang::CastExpr const *cast) {
    auto const loc = cast->getBeginLoc();
    auto const *sub = cast->getSubExpr();
    auto const target = typeName(cast->getType(), loc);
    if (isLiteralLike(sub)) {
        auto value = clang::Expr::EvalResult{};
        if (cast->EvaluateAsRValue(value, context)) {
            return constantValue(value.Val, cast->getType(), loc);
        }
    }
    auto const inner = expr(sub);
    if (typeName(sub->getType(), loc) == target and llvm::isa<clang::ImplicitCastExpr>(cast)) {
        return inner;
    }
    return target + "(" + inner + ")";
}

std::string Writer::binary(clang::BinaryOperator const *op) {
    auto const opcode = op->getOpcode();
    auto const spelling = clang::BinaryOperator::getOpcodeStr(opcode).str();
    if (opcode == clang::BO_PtrMemD or opcode == clang::BO_PtrMemI) {
        unsupported(op->getBeginLoc(), "pointer-to-member access");
        return "0";
    }
    if (opcode == clang::BO_Comma) {
        return expr(op->getLHS()) + ", " + expr(op->getRHS());
    }
    if (opcode == clang::BO_Assign) {
        auto const lhs = exprAs(op->getLHS(), Role::Write);
        return lhs + " = " + expr(op->getRHS());
    }
    if (auto const *compound = llvm::dyn_cast<clang::CompoundAssignOperator>(op)) {
        auto const loc = op->getBeginLoc();
        auto const lhsType = typeName(op->getLHS()->getType(), loc);
        auto const computeType = typeName(compound->getComputationLHSType(), loc);
        if (lhsType == computeType) {
            auto const lhs = exprAs(op->getLHS(), Role::ReadWrite);
            return lhs + " " + spelling + " " + expr(op->getRHS());
        }
        // `int i; i += 1.5f;` computes in float: GLSL needs it spelled out.
        if (op->getLHS()->HasSideEffects(context)) {
            diags.error(loc, "this mixed-type compound assignment needs a side-effect free left-hand side");
        }
        auto const lhs = exprAs(op->getLHS(), Role::ReadWrite);
        auto const plain = spelling.substr(0, spelling.size() - 1);
        return lhs + " = " + lhsType + "(" + computeType + "(" + lhs + ") " + plain + " (" + expr(op->getRHS()) + "))";
    }
    return expr(op->getLHS()) + " " + spelling + " " + expr(op->getRHS());
}

std::string Writer::unary(clang::UnaryOperator const *op, Role const /*inherited*/) {
    auto const opcode = op->getOpcode();
    switch (opcode) {
    case clang::UO_AddrOf:
    case clang::UO_Deref:
        unsupported(op->getBeginLoc(), "taking addresses or dereferencing pointers");
        return "0";
    case clang::UO_PostInc:
    case clang::UO_PostDec:
        return exprAs(op->getSubExpr(), Role::ReadWrite) + clang::UnaryOperator::getOpcodeStr(opcode).str();
    case clang::UO_PreInc:
    case clang::UO_PreDec:
        return clang::UnaryOperator::getOpcodeStr(opcode).str() + exprAs(op->getSubExpr(), Role::ReadWrite);
    case clang::UO_Plus:
    case clang::UO_Minus:
    case clang::UO_Not:
    case clang::UO_LNot:
        return clang::UnaryOperator::getOpcodeStr(opcode).str() + expr(op->getSubExpr());
    case clang::UO_Extension:
        return expr(op->getSubExpr());
    default:
        unsupported(op->getBeginLoc(), "this unary operator");
        return "0";
    }
}

std::string Writer::declRef(clang::DeclRefExpr const *ref, Role const inherited) {
    auto const *decl = ref->getDecl();
    auto const loc = ref->getBeginLoc();
    if (auto const *var = llvm::dyn_cast<clang::VarDecl>(decl)) {
        if (var->hasLocalStorage() or llvm::isa<clang::ParmVarDecl>(var)) {
            return localName(var);
        }
        if (isTcDecl(var)) {
            if (inherited != Role::Read) {
                diags.error(loc, "'" + var->getName().str() + "' is a read-only GLSL input");
            }
            return var->getName().str();
        }
        if (inherited != Role::Read) {
            diags.error(loc, "global '" + var->getName().str() + "' is a GLSL constant and cannot be modified");
        }
        return requireGlobalConstant(var, loc);
    }
    if (auto const *enumerator = llvm::dyn_cast<clang::EnumConstantDecl>(decl)) {
        auto const isUnsigned = typeName(ref->getType(), loc) == "uint";
        auto const &value = enumerator->getInitVal();
        return isUnsigned ? std::to_string(value.getZExtValue()) + "u" : std::to_string(value.getSExtValue());
    }
    unsupported(loc, "reference to '" + decl->getNameAsString() + "'");
    return "0";
}

clang::FieldDecl const *Writer::kernelField(clang::Expr const *e) const {
    auto const *me = llvm::dyn_cast<clang::MemberExpr>(e->IgnoreParenImpCasts());
    if (me == nullptr or not llvm::isa<clang::CXXThisExpr>(me->getBase()->IgnoreParenImpCasts())) {
        return nullptr;
    }
    auto const *field = llvm::dyn_cast<clang::FieldDecl>(me->getMemberDecl());
    return field != nullptr and field->getParent() == kernel.record ? field : nullptr;
}

std::string Writer::kernelMember(clang::FieldDecl const *field, Role const inherited, clang::SourceLocation const loc) {
    if (kernel.findBuffer(field) != nullptr or kernel.findTable(field) != nullptr or kernel.findImage(field) != nullptr
        or kernel.isTexture(field) or kernel.isShared(field)) {
        return glslIdentifier(field->getName());
    }
    if (kernel.findUniform(field) != nullptr) {
        if (inherited != Role::Read) {
            diags.error(loc, "uniform '" + field->getName().str() + "' is read-only inside the kernel");
        }
        return glslIdentifier(field->getName());
    }
    if (inherited != Role::Read) {
        diags.error(loc, "kernel member '" + field->getName().str()
                             + "' is a constant on the GPU and cannot be modified; use a local variable");
    }
    return requireKernelConstant(field);
}

std::string Writer::member(clang::MemberExpr const *me, Role const inherited) {
    auto const loc = me->getBeginLoc();
    if (me == replacement.node) {
        return replacement.name;
    }
    if (auto const access = tableFieldAccess(me)) {
        return tableRead(*access, inherited, loc);
    }
    auto const *base = me->getBase();
    auto const baseIsThis = llvm::isa<clang::CXXThisExpr>(base->IgnoreParenImpCasts());
    if (auto const *var = llvm::dyn_cast<clang::VarDecl>(me->getMemberDecl())) {
        if (inherited != Role::Read) {
            diags.error(loc, "static member '" + var->getName().str() + "' cannot be modified in a kernel");
        }
        return requireGlobalConstant(var, loc);
    }
    auto const *field = llvm::dyn_cast<clang::FieldDecl>(me->getMemberDecl());
    if (field == nullptr) {
        unsupported(loc, "member function reference outside a call");
        return "0";
    }
    if (me->isArrow() and not baseIsThis) {
        unsupported(loc, "pointer member access");
        return "0";
    }
    if (baseIsThis) {
        if (inKernelScope() and field->getParent() == kernel.record) {
            return kernelMember(field, inherited, loc);
        }
        if (currentKind == FunctionKind::StructMethod) {
            return "self." + fieldName(field);
        }
        unsupported(loc, "this member access");
        return "0";
    }
    if (asTcSpecialization(field->getParent(), "Uniform") != nullptr
        or asTcSpecialization(field->getParent(), "Shared") != nullptr
        or asTcSpecialization(field->getParent(), "Encoded") != nullptr) {
        return exprAs(base, inherited);
    }
    auto const object = exprAs(base, inherited);
    if (asTcSpecialization(field->getParent(), "detail::VecFields") != nullptr) {
        return object + "." + field->getName().str();
    }
    if (isTcDecl(field->getParent())) {
        unsupported(loc, "access to '" + field->getName().str() + "'");
        return "0";
    }
    return object + "." + fieldName(field);
}

// GLSL atomics operate on buffer or shared memory only: a buffer element (or
// a member of one), or a Shared variable (or an element of one).
bool Writer::isAtomicTarget(clang::Expr const *e) const {
    e = e->IgnoreParenImpCasts();
    if (auto const *me = llvm::dyn_cast<clang::MemberExpr>(e)) {
        auto const *field = llvm::dyn_cast<clang::FieldDecl>(me->getMemberDecl());
        if (field != nullptr and asTcSpecialization(field->getParent(), "Shared") != nullptr) {
            return true;
        }
        return not llvm::isa<clang::CXXThisExpr>(me->getBase()->IgnoreParenImpCasts()) and isAtomicTarget(me->getBase());
    }
    if (auto const *oc = llvm::dyn_cast<clang::CXXOperatorCallExpr>(e); oc != nullptr and oc->getOperator() == clang::OO_Subscript) {
        auto const objectType = bareType(oc->getArg(0)->getType());
        if (asTcSpecialization(objectType, "BufferBinding") != nullptr or asTcSpecialization(objectType, "Shared") != nullptr) {
            return true;
        }
        return (asStdArray(objectType) or asTcVec(objectType)) and isAtomicTarget(oc->getArg(0));
    }
    if (auto const *subscript = llvm::dyn_cast<clang::ArraySubscriptExpr>(e)) {
        return isAtomicTarget(subscript->getBase());
    }
    return false;
}

void Writer::markBufferUse(clang::Expr const *object, Role const use) {
    auto const *field = kernelField(object);
    auto *buffer = field != nullptr ? kernel.findBuffer(field) : nullptr;
    if (buffer == nullptr) {
        return;
    }
    buffer->usage.read = buffer->usage.read or use != Role::Write;
    buffer->usage.written = buffer->usage.written or use != Role::Read;
}

void Writer::markImageUse(clang::Expr const *image, bool const write) {
    auto const *field = kernelField(image);
    auto *model = field != nullptr ? kernel.findImage(field) : nullptr;
    if (model == nullptr) {
        diags.error(image->getBeginLoc(), "image functions need an ImageBinding member of the kernel");
        return;
    }
    (write ? model->usage.written : model->usage.read) = true;
}

std::string Writer::arguments(clang::FunctionDecl const *callee, llvm::ArrayRef<clang::Expr const *> const args) {
    auto parts = std::vector<std::string>{};
    for (std::size_t i = 0; i < args.size(); ++i) {
        auto const byReference = i < callee->getNumParams()
                             and isNonConstLValueRef(callee->getParamDecl(static_cast<unsigned>(i))->getType());
        parts.push_back(exprAs(args[i], byReference ? Role::ReadWrite : Role::Read));
    }
    return join(parts, ", ");
}

std::string Writer::memberCall(clang::CXXMemberCallExpr const *mc, Role const inherited) {
    auto const loc = mc->getBeginLoc();
    auto const *method = mc->getMethodDecl();
    auto const *record = method->getParent();
    auto const *object = mc->getImplicitObjectArgument();
    auto const name = method->getDeclName().isIdentifier() ? method->getName() : llvm::StringRef{};
    auto const args = llvm::ArrayRef<clang::Expr const *>{mc->getArgs(), mc->getNumArgs()};

    if (llvm::isa<clang::CXXConversionDecl>(method)
        and (asTcSpecialization(record, "Uniform") != nullptr or asTcSpecialization(record, "Shared") != nullptr
             or asTcSpecialization(record, "Encoded") != nullptr)) {
        return exprAs(object, inherited);
    }
    if (asTcSpecialization(record, "detail::VecFields") != nullptr and isSwizzleName(name)) {
        return expr(object) + "." + name.str();
    }
    if (asTcSpecialization(record, "BufferBinding") != nullptr and name == "length") {
        return expr(object) + ".length()";
    }
    if (asTcSpecialization(record, "TableBinding") != nullptr and name == "length") {
        auto const *field = kernelField(object);
        auto *table = field != nullptr ? kernel.findTable(field) : nullptr;
        if (table == nullptr) {
            unsupported(loc, "a TableBinding that is not a member of the kernel");
            return "0";
        }
        table->usage.read = true;
        return "int(" + table->name + "[0])"; // header word 0: the element count
    }
    if (auto const array = asStdArray(context.getRecordType(record))) {
        if (name == "size") {
            return std::to_string(array->size) + "u";
        }
        unsupported(loc, "std::array::" + name.str());
        return "0";
    }
    if (isTcDecl(record)) {
        unsupported(loc, "'" + method->getQualifiedNameAsString() + "'");
        return "0";
    }
    if (record == kernel.record) {
        if (not llvm::isa<clang::CXXThisExpr>(object->IgnoreParenImpCasts())) {
            unsupported(loc, "calling a member function of another kernel object");
            return "0";
        }
        return requireFunction(method, FunctionKind::KernelMethod, loc) + "(" + arguments(method, args) + ")";
    }
    auto const fn = requireFunction(method, FunctionKind::StructMethod, loc);
    if (not method->isConst() and not object->isLValue()) {
        diags.error(loc, "non-const member functions can only be called on variables in kernels");
    }
    auto const self = exprAs(object, method->isConst() ? Role::Read : Role::ReadWrite);
    auto const rest = arguments(method, args);
    return fn + "(" + self + (rest.empty() ? "" : ", " + rest) + ")";
}

std::string Writer::operatorCall(clang::CXXOperatorCallExpr const *oc, Role const inherited) {
    auto const loc = oc->getBeginLoc();
    auto const op = oc->getOperator();
    auto const *callee = oc->getDirectCallee();
    if (op == clang::OO_Subscript) {
        auto const *object = oc->getArg(0);
        auto const objectType = bareType(object->getType());
        if (asTcSpecialization(objectType, "TableBinding") != nullptr) {
            if (oc == replacement.node) {
                return replacement.name;
            }
            auto *table = tableOfSubscript(oc);
            if (table == nullptr) {
                unsupported(loc, "a TableBinding that is not a member of the kernel");
                return "0";
            }
            return tableRead(TableAccess{table, oc->getArg(1), nullptr, oc}, inherited, loc);
        }
        if (asTcSpecialization(objectType, "BufferBinding") != nullptr) {
            markBufferUse(object, inherited);
            auto const name = expr(object);
            return name + "[" + expr(oc->getArg(1)) + "]";
        }
        if (asTcVec(objectType) or asTcMat(objectType) or asStdArray(objectType)
            or asTcSpecialization(objectType, "Shared") != nullptr) {
            auto const base = exprAs(object, inherited);
            return base + "[" + expr(oc->getArg(1)) + "]";
        }
        if (callee == nullptr or isTcDecl(callee) or callee->isInStdNamespace()) {
            unsupported(loc, "operator[] of '" + objectType.getAsString() + "'");
            return "0";
        }
    }
    auto const *method = llvm::dyn_cast_or_null<clang::CXXMethodDecl>(callee);
    auto const isImplicitAssignment = method != nullptr
                                  and (method->isCopyAssignmentOperator() or method->isMoveAssignmentOperator())
                                  and (method->isImplicit() or method->isDefaulted());
    if (callee == nullptr) {
        unsupported(loc, "calling an operator through a pointer");
        return "0";
    }
    // Operators of user types become calls of their translated functions.
    if (not isTcDecl(callee) and not isImplicitAssignment) {
        auto const args = llvm::ArrayRef<clang::Expr const *>{oc->getArgs(), oc->getNumArgs()};
        if (method != nullptr and not method->isStatic()) {
            auto const fn = requireFunction(method, FunctionKind::StructMethod, loc);
            if (not method->isConst() and not args[0]->isLValue()) {
                diags.error(loc, "non-const member operators can only be applied to variables in kernels");
            }
            auto const self = exprAs(args[0], method->isConst() ? Role::Read : Role::ReadWrite);
            auto const rest = arguments(method, args.drop_front());
            return fn + "(" + self + (rest.empty() ? "" : ", " + rest) + ")";
        }
        return requireFunction(callee, FunctionKind::Free, loc) + "(" + arguments(callee, args) + ")";
    }
    auto const spelling = std::string{clang::getOperatorSpelling(op)};
    switch (op) {
    case clang::OO_Equal: {
        auto const lhs = exprAs(oc->getArg(0), Role::Write);
        return lhs + " = " + expr(oc->getArg(1));
    }
    case clang::OO_PlusEqual:
    case clang::OO_MinusEqual:
    case clang::OO_StarEqual:
    case clang::OO_SlashEqual:
    case clang::OO_PercentEqual:
    case clang::OO_AmpEqual:
    case clang::OO_PipeEqual:
    case clang::OO_CaretEqual:
    case clang::OO_LessLessEqual:
    case clang::OO_GreaterGreaterEqual: {
        auto const lhs = exprAs(oc->getArg(0), Role::ReadWrite);
        return lhs + " " + spelling + " " + expr(oc->getArg(1));
    }
    case clang::OO_PlusPlus:
    case clang::OO_MinusMinus:
        if (oc->getNumArgs() == 2) {
            return exprAs(oc->getArg(0), Role::ReadWrite) + spelling;
        }
        return spelling + exprAs(oc->getArg(0), Role::ReadWrite);
    case clang::OO_Call:
    case clang::OO_Arrow:
    case clang::OO_ArrowStar:
    case clang::OO_Comma:
        unsupported(loc, "operator" + spelling);
        return "0";
    default:
        if (oc->getNumArgs() == 1) {
            return spelling + expr(oc->getArg(0));
        }
        return expr(oc->getArg(0)) + " " + spelling + " " + expr(oc->getArg(1));
    }
}

std::string Writer::call(clang::CallExpr const *c) {
    auto const loc = c->getBeginLoc();
    auto const *callee = c->getDirectCallee();
    if (callee == nullptr) {
        unsupported(loc, "calling through a function pointer");
        return "0";
    }
    auto const args = llvm::ArrayRef<clang::Expr const *>{c->getArgs(), c->getNumArgs()};
    if (auto const *method = llvm::dyn_cast<clang::CXXMethodDecl>(callee); method != nullptr and method->isStatic()) {
        auto const recordType = context.getRecordType(method->getParent());
        if (auto const vec = asTcVec(recordType); vec and method->getName() == "length") {
            return std::to_string(vec->size);
        }
        if (auto const matrix = asTcMat(recordType); matrix and method->getName() == "length") {
            return std::to_string(matrix->columns);
        }
        if (isTcDecl(method)) {
            unsupported(loc, "'" + method->getQualifiedNameAsString() + "'");
            return "0";
        }
        return requireFunction(method, FunctionKind::Free, loc) + "(" + arguments(method, args) + ")";
    }
    auto const name = callee->getDeclName().isIdentifier() ? callee->getName() : llvm::StringRef{};
    if (isTcDecl(callee)) {
        if (name == "imageLoad") {
            markImageUse(c->getArg(0), false);
        } else if (name == "imageStore") {
            markImageUse(c->getArg(0), true);
        } else if (name == "barrier" and not kernel.usesBarrier) {
            diags.error(loc, "kernels that call barrier() must declare 'static constexpr bool usesBarrier = true;' "
                             "so the CPU backend can synchronise their invocations");
        } else if (name.starts_with("atomic") and c->getNumArgs() > 0 and not isAtomicTarget(c->getArg(0))) {
            diags.error(c->getArg(0)->getBeginLoc(),
                        name.str() + " needs a buffer element or a Shared variable as its first argument");
        }
        return glslNameOfTcFunction(name) + "(" + arguments(callee, args) + ")";
    }
    auto const &sources = context.getSourceManager();
    if (callee->isInStdNamespace() or sources.isInSystemHeader(callee->getLocation())) {
        if (auto const glslName = glslNameOfStdFunction(name)) {
            return *glslName + "(" + arguments(callee, args) + ")";
        }
        unsupported(loc, "'" + callee->getQualifiedNameAsString() + "'");
        return "0";
    }
    return requireFunction(callee, FunctionKind::Free, loc) + "(" + arguments(callee, args) + ")";
}

std::string Writer::construct(clang::CXXConstructExpr const *ctor, Role const inherited) {
    auto const loc = ctor->getBeginLoc();
    auto const *decl = ctor->getConstructor();
    auto const type = ctor->getType();
    if (decl->isCopyOrMoveConstructor() and ctor->getNumArgs() >= 1) {
        return exprAs(ctor->getArg(0), inherited);
    }
    if (asTcSpecialization(type, "Encoded") != nullptr) {
        return ctor->getNumArgs() == 1 ? expr(ctor->getArg(0)) : zeroValue(type, loc);
    }
    if (asTcVec(type) or asTcMat(type)) {
        if (ctor->getNumArgs() == 0) {
            return zeroValue(type, loc);
        }
        auto parts = std::vector<std::string>{};
        for (auto const *arg : ctor->arguments()) {
            parts.push_back(expr(arg));
        }
        return typeName(type, loc) + "(" + join(parts, ", ") + ")";
    }
    auto const *record = bareType(type)->getAsCXXRecordDecl();
    if (ctor->getNumArgs() == 0 and record != nullptr and not isTcDecl(record)) {
        return zeroValue(type, loc);
    }
    unsupported(loc, "constructing '" + type.getAsString() + "'");
    return "0";
}

std::string Writer::arrayInit(clang::InitListExpr const *list, clang::QualType const element, std::uint64_t const size) {
    auto parts = std::vector<std::string>{};
    for (std::uint64_t i = 0; i < size; ++i) {
        if (i < list->getNumInits()) {
            parts.push_back(expr(list->getInit(static_cast<unsigned>(i))));
        } else if (list->hasArrayFiller()) {
            parts.push_back(expr(list->getArrayFiller()));
        } else {
            parts.push_back(zeroValue(element, list->getBeginLoc()));
        }
    }
    return typeName(element, list->getBeginLoc()) + "[" + std::to_string(size) + "](" + join(parts, ", ") + ")";
}

std::string Writer::initList(clang::InitListExpr const *list) {
    if (not list->isSemanticForm()) {
        list = list->getSemanticForm();
    }
    auto const loc = list->getBeginLoc();
    auto const type = bareType(list->getType());
    if (auto const array = asStdArray(type)) {
        if (list->getNumInits() == 0) {
            return zeroValue(type, loc);
        }
        auto const *inner = list->getInit(0);
        if (auto const *innerList = llvm::dyn_cast<clang::InitListExpr>(inner)) {
            if (not innerList->isSemanticForm()) {
                innerList = innerList->getSemanticForm();
            }
            return arrayInit(innerList, array->element, array->size);
        }
        return expr(inner);
    }
    if (auto const *carray = context.getAsConstantArrayType(type)) {
        return arrayInit(list, carray->getElementType(), carray->getSize().getZExtValue());
    }
    if (auto const *record = type->getAsCXXRecordDecl()) {
        auto const name = requireStruct(record, loc);
        auto parts = std::vector<std::string>{};
        for (unsigned i = 0; i < list->getNumInits(); ++i) {
            parts.push_back(expr(list->getInit(i)));
        }
        return name + "(" + join(parts, ", ") + ")";
    }
    if (list->getNumInits() == 1) {
        return expr(list->getInit(0));
    }
    if (list->getNumInits() == 0) {
        return zeroValue(type, loc);
    }
    unsupported(loc, "this initializer list");
    return "0";
}

// ---------------------------------------------------------------- tables

namespace {

// `t[i]` may reach a member access or an assignment through temporaries and
// no-op casts.
clang::Expr const *stripTableWrappers(clang::Expr const *e) {
    for (;;) {
        if (auto const *paren = llvm::dyn_cast<clang::ParenExpr>(e)) {
            e = paren->getSubExpr();
        } else if (auto const *cast = llvm::dyn_cast<clang::ImplicitCastExpr>(e);
                   cast != nullptr
                   and (cast->getCastKind() == clang::CK_NoOp or cast->getCastKind() == clang::CK_LValueToRValue
                        or cast->getCastKind() == clang::CK_DerivedToBase
                        or cast->getCastKind() == clang::CK_UncheckedDerivedToBase)) {
            e = cast->getSubExpr();
        } else if (auto const *temp = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(e)) {
            e = temp->getSubExpr();
        } else if (auto const *bind = llvm::dyn_cast<clang::CXXBindTemporaryExpr>(e)) {
            e = bind->getSubExpr();
        } else if (auto const *full = llvm::dyn_cast<clang::FullExpr>(e)) {
            e = full->getSubExpr();
        } else {
            return e;
        }
    }
}

bool isCompoundAssignment(clang::OverloadedOperatorKind const op) {
    switch (op) {
    case clang::OO_PlusEqual:
    case clang::OO_MinusEqual:
    case clang::OO_StarEqual:
    case clang::OO_SlashEqual:
    case clang::OO_PercentEqual:
    case clang::OO_AmpEqual:
    case clang::OO_PipeEqual:
    case clang::OO_CaretEqual:
    case clang::OO_LessLessEqual:
    case clang::OO_GreaterGreaterEqual:
        return true;
    default:
        return false;
    }
}

} // namespace

TableModel *Writer::tableOfSubscript(clang::CXXOperatorCallExpr const *call) {
    if (call->getOperator() != clang::OO_Subscript
        or asTcSpecialization(bareType(call->getArg(0)->getType()), "TableBinding") == nullptr) {
        return nullptr;
    }
    auto const *field = kernelField(call->getArg(0));
    return field != nullptr ? kernel.findTable(field) : nullptr;
}

// `t[i].field` for a top-level field of a struct element.
std::optional<Writer::TableAccess> Writer::tableFieldAccess(clang::MemberExpr const *member) {
    auto const *field = llvm::dyn_cast<clang::FieldDecl>(member->getMemberDecl());
    auto const *call = llvm::dyn_cast<clang::CXXOperatorCallExpr>(stripTableWrappers(member->getBase()));
    if (field == nullptr or call == nullptr) {
        return std::nullopt;
    }
    auto *table = tableOfSubscript(call);
    if (table == nullptr or not table->layout.isRecord) {
        return std::nullopt;
    }
    for (auto const &packed : table->layout.fields) {
        if (packed.decl->getCanonicalDecl() == field->getCanonicalDecl()) {
            return TableAccess{table, call->getArg(1), &packed, member};
        }
    }
    return std::nullopt;
}

// The table access an lvalue such as `t[i].velocity.y` or `t[i].weights[2]`
// lives in: the top-level field, or the whole element.
std::optional<Writer::TableAccess> Writer::tableTarget(clang::Expr const *e) {
    for (auto const *node = e;;) {
        node = stripTableWrappers(node);
        if (auto const *member = llvm::dyn_cast<clang::MemberExpr>(node)) {
            if (auto const access = tableFieldAccess(member)) {
                return access;
            }
            if (llvm::isa<clang::CXXThisExpr>(member->getBase()->IgnoreParenImpCasts())) {
                return std::nullopt;
            }
            node = member->getBase();
        } else if (auto const *call = llvm::dyn_cast<clang::CXXOperatorCallExpr>(node);
                   call != nullptr and call->getOperator() == clang::OO_Subscript) {
            if (auto *table = tableOfSubscript(call)) {
                return TableAccess{table, call->getArg(1), nullptr, call};
            }
            node = call->getArg(0);
        } else if (auto const *subscript = llvm::dyn_cast<clang::ArraySubscriptExpr>(node)) {
            node = subscript->getBase();
        } else {
            return std::nullopt;
        }
    }
}

std::string Writer::tableRead(TableAccess const &access, Role const inherited, clang::SourceLocation const loc) {
    if (inherited != Role::Read) {
        diags.error(loc, "a table element can only be written by a statement of its own, such as `t[i] = v;`, "
                         "`t[i].field = v;` or `t[i].field.x += v;`; pass a copy to functions that modify it");
        return "0";
    }
    access.table->usage.read = true;
    return tableGetter(*access.table, access.field) + "(" + tableIndex(access.index) + ")";
}

std::string Writer::tableIndex(clang::Expr const *index) {
    auto const text = expr(index);
    return typeName(index->getType(), index->getBeginLoc()) == "uint" ? text : "uint(" + text + ")";
}

// `t[i] = v;` and `t[i].field = v;` store directly. Any other write loads
// the field (or element) into a local, applies the statement to it and
// stores it back.
std::optional<std::string> Writer::tableWriteStatement(clang::Expr const *e, int const depth) {
    auto const *full = stripTableWrappers(e);
    auto const *lhs = static_cast<clang::Expr const *>(nullptr);
    auto const *rhs = static_cast<clang::Expr const *>(nullptr);
    auto plain = false;
    if (auto const *op = llvm::dyn_cast<clang::BinaryOperator>(full); op != nullptr and op->isAssignmentOp()) {
        lhs = op->getLHS();
        rhs = op->getRHS();
        plain = op->getOpcode() == clang::BO_Assign;
    } else if (auto const *unary = llvm::dyn_cast<clang::UnaryOperator>(full);
               unary != nullptr and unary->isIncrementDecrementOp()) {
        lhs = unary->getSubExpr();
    } else if (auto const *call = llvm::dyn_cast<clang::CXXOperatorCallExpr>(full); call != nullptr) {
        auto const op = call->getOperator();
        if (op == clang::OO_Equal) {
            lhs = call->getArg(0);
            rhs = call->getArg(1);
            plain = true;
        } else if (isCompoundAssignment(op) or op == clang::OO_PlusPlus or op == clang::OO_MinusMinus) {
            lhs = call->getArg(0);
        }
    }
    if (lhs == nullptr) {
        return std::nullopt;
    }
    auto const access = tableTarget(lhs);
    if (not access) {
        return std::nullopt;
    }
    auto const loc = e->getBeginLoc();
    if (access->index->HasSideEffects(context)) {
        diags.error(access->index->getBeginLoc(), "the index of a table element that is written must not have "
                                                  "side effects; compute it into a local first");
    }
    auto &table = *access->table;
    table.usage.written = true;
    auto const pad = indent(depth);
    auto const index = tableIndex(access->index);
    if (plain and stripTableWrappers(lhs) == access->node) {
        return pad + tableSetter(table, access->field) + "(" + index + ", " + expr(rhs) + ");\n";
    }
    table.usage.read = true;
    auto const temp = "tc_t" + std::to_string(loopCounter++);
    auto const type = access->field != nullptr ? access->field->type : table.elementType;
    auto text = pad + "{\n";
    text += indent(depth + 1) + declaration(type, temp, loc) + " = " + tableGetter(table, access->field) + "(" + index
          + ");\n";
    auto const saved = std::exchange(replacement, Replacement{access->node, temp});
    text += indent(depth + 1) + expr(e) + ";\n";
    replacement = saved;
    text += indent(depth + 1) + tableSetter(table, access->field) + "(" + index + ", " + temp + ");\n";
    return text + pad + "}\n";
}

namespace {

std::string uintLiteral(std::uint64_t const value) {
    return std::to_string(value) + "u";
}

} // namespace

// The words of element i an accessor moves start at 4 + fieldAddress(...)
// (after the four-word header), computed by the layout's own fieldAddress
// translated to GLSL. With contiguous records that is the whole element and
// fields are word ranges of it; otherwise it is the field's own words. The
// layout's fieldAlignment, evaluated now, says which chunks never straddle
// elements.
Writer::TableWindow Writer::tableWindow(TableModel &table, PackedField const *field) {
    auto const loc = table.field->getLocation();
    auto const &addressing = table.addressing;
    auto const words = table.layout.words;
    auto const contiguous = addressing.contiguousRecords;
    auto const offset = contiguous or field == nullptr ? 0u : field->offset;
    auto const length = contiguous or field == nullptr ? words : field->words;
    auto alignment = evaluateCall(context, addressing.fieldAlignment, {words, offset, length});
    if (not alignment or (*alignment != 1 and *alignment != 2 and *alignment != 4)) {
        diags.warning(loc, "fieldAlignment of the layout of '" + table.name
                               + "' did not evaluate to 1, 2 or 4 at build time; its words are moved one by one");
        alignment = 1;
    }
    auto const address = requireFunction(addressing.fieldAddress, FunctionKind::Free, loc) + "(i, "
                       + uintLiteral(words) + ", " + uintLiteral(offset) + ", " + uintLiteral(length) + ")";
    auto window = TableWindow{};
    window.vector = static_cast<std::uint32_t>(*alignment);
    window.chunkBase = window.vector == 1 ? "4u + " + address : "(4u + " + address + ") / " + uintLiteral(window.vector);
    window.first = contiguous and field != nullptr ? field->offset : 0;
    window.last = contiguous and field != nullptr ? field->offset + field->words : length;
    return window;
}

std::string Writer::tableView(TableModel &table, std::uint32_t const vector) {
    if (vector == 2) {
        table.usesView2 = true;
        return table.name + "_2";
    }
    if (vector == 4) {
        table.usesView4 = true;
        return table.name + "_4";
    }
    return table.name;
}

// Loads the chunks covering the window and unpacks the value from them.
std::string Writer::tableGetter(TableModel &table, PackedField const *field) {
    auto const loc = table.field->getLocation();
    auto const suffix = field != nullptr and field->decl != nullptr ? "_" + fieldName(field->decl) : std::string{};
    auto const name = "tc_" + table.name + "_get" + suffix;
    if (definedAccessors.count(name) != 0) {
        return name;
    }
    auto const type = field != nullptr ? field->type : table.elementType;
    if (field == nullptr and table.layout.isRecord and not table.addressing.contiguousRecords) {
        auto parts = std::vector<std::string>{};
        for (auto const &packed : table.layout.fields) {
            parts.push_back(tableGetter(table, &packed) + "(i)");
        }
        auto const typeText = typeName(type, loc);
        definedAccessors.insert(name);
        out.tableAccessors.push_back(typeText + " " + name + "(uint i) {\n    return " + typeText + "("
                                     + join(parts, ", ") + ");\n}");
        return name;
    }
    auto const window = tableWindow(table, field);
    auto const view = tableView(table, window.vector);
    auto const chunkType = window.vector == 1 ? std::string{"uint"} : "uvec" + std::to_string(window.vector);
    auto body = "    uint b = " + window.chunkBase + ";\n";
    auto words = std::vector<std::string>{};
    for (auto chunk = window.first / window.vector; chunk * window.vector < window.last; ++chunk) {
        auto const name_ = "c" + std::to_string(chunk);
        body += "    " + chunkType + " " + name_ + " = " + view + "[b + " + uintLiteral(chunk) + "];\n";
        for (std::uint32_t k = 0; k < window.vector; ++k) {
            auto const word = chunk * window.vector + k;
            if (word >= window.first and word < window.last) {
                words.push_back(window.vector == 1 ? name_ : name_ + "." + "xyzw"[k]);
            }
        }
    }
    auto cursor = std::uint32_t{0};
    body += "    return " + unpackValue(type, words, cursor, loc) + ";\n";
    definedAccessors.insert(name);
    out.tableAccessors.push_back(typeName(type, loc) + " " + name + "(uint i) {\n" + body + "}");
    return name;
}

// Stores through the same uint / uvec2 / uvec4 view the getter uses. A
// partial chunk is a load/modify/store of that vector; writing the words
// through uint[] while the getter reads uvec4[] is not aliasing on some
// Vulkan compilers, and the next load sees the old value.
std::string Writer::tableSetter(TableModel &table, PackedField const *field) {
    auto const loc = table.field->getLocation();
    auto const suffix = field != nullptr and field->decl != nullptr ? "_" + fieldName(field->decl) : std::string{};
    auto const name = "tc_" + table.name + "_set" + suffix;
    if (definedAccessors.count(name) != 0) {
        return name;
    }
    auto const type = field != nullptr ? field->type : table.elementType;
    if (field == nullptr and table.layout.isRecord and not table.addressing.contiguousRecords) {
        auto body = std::string{};
        for (auto const &packed : table.layout.fields) {
            body += "    " + tableSetter(table, &packed) + "(i, v." + fieldName(packed.decl) + ");\n";
        }
        definedAccessors.insert(name);
        out.tableAccessors.push_back("void " + name + "(uint i, " + declaration(type, "v", loc) + ") {\n" + body
                                     + "}");
        return name;
    }
    auto const window = tableWindow(table, field);
    auto words = std::vector<std::string>{};
    packWords(type, "v", words, loc);
    auto const word = [&](std::uint32_t const index) { return words[index - window.first]; };
    auto const view = tableView(table, window.vector);
    auto body = "    uint b = " + window.chunkBase + ";\n";
    for (auto chunk = window.first / window.vector; chunk * window.vector < window.last; ++chunk) {
        auto const begin = chunk * window.vector;
        auto const end = begin + window.vector;
        auto const slot = view + "[b + " + uintLiteral(chunk) + "]";
        if (window.vector == 1) {
            body += "    " + slot + " = " + word(begin) + ";\n";
            continue;
        }
        if (begin >= window.first and end <= window.last) {
            auto parts = std::vector<std::string>{};
            for (auto w = begin; w < end; ++w) {
                parts.push_back(word(w));
            }
            body += "    " + slot + " = uvec" + std::to_string(window.vector) + "(" + join(parts, ", ") + ");\n";
            continue;
        }
        auto const temp = "c" + std::to_string(chunk);
        body += "    uvec" + std::to_string(window.vector) + " " + temp + " = " + slot + ";\n";
        for (auto w = std::max(begin, window.first); w < std::min(end, window.last); ++w) {
            body += "    " + temp + "." + "xyzw"[w - begin] + " = " + word(w) + ";\n";
        }
        body += "    " + slot + " = " + temp + ";\n";
    }
    definedAccessors.insert(name);
    out.tableAccessors.push_back("void " + name + "(uint i, " + declaration(type, "v", loc) + ") {\n" + body + "}");
    return name;
}

// A GLSL expression rebuilding a value from `words` (uint expressions),
// consumed from `cursor` on.
std::string Writer::unpackValue(clang::QualType const original, std::vector<std::string> const &words,
                                std::uint32_t &cursor, clang::SourceLocation const loc) {
    if (auto const encoded = encodedInfo(original, context, diags, loc)) {
        auto parts = std::vector<std::string>{};
        for (std::uint32_t i = 0; i < encoded->words; ++i) {
            parts.push_back(words[cursor++]);
        }
        return requireFunction(encoded->decode, FunctionKind::Free, loc) + "(uint[" + std::to_string(encoded->words)
             + "](" + join(parts, ", ") + "))";
    }
    auto type = bareType(original);
    if (auto const *enumType = type->getAs<clang::EnumType>()) {
        type = bareType(enumType->getDecl()->getIntegerType());
    }
    auto const word = [&] { return words[cursor++]; };
    if (type->isBooleanType()) {
        return "(" + word() + " != 0u)";
    }
    if (type->isRealFloatingType()) {
        if (typeName(type, loc) == "double") {
            auto const low = word();
            auto const high = word();
            return "packDouble2x32(uvec2(" + low + ", " + high + "))";
        }
        return "uintBitsToFloat(" + word() + ")";
    }
    if (type->isIntegerType()) {
        return typeName(type, loc) == "uint" ? word() : "int(" + word() + ")";
    }
    auto parts = std::vector<std::string>{};
    if (auto const vec = asTcVec(type)) {
        for (unsigned i = 0; i < vec->size; ++i) {
            parts.push_back(unpackValue(vec->element, words, cursor, loc));
        }
        return typeName(type, loc) + "(" + join(parts, ", ") + ")";
    }
    if (auto const matrix = asTcMat(type)) {
        for (unsigned i = 0; i < matrix->columns * matrix->rows; ++i) {
            parts.push_back(unpackValue(matrix->element, words, cursor, loc));
        }
        return typeName(type, loc) + "(" + join(parts, ", ") + ")";
    }
    if (auto const array = asStdArray(type)) {
        for (std::uint64_t i = 0; i < array->size; ++i) {
            parts.push_back(unpackValue(array->element, words, cursor, loc));
        }
        return typeName(type, loc) + "(" + join(parts, ", ") + ")";
    }
    auto const *record = type->getAsCXXRecordDecl();
    for (auto const *field : record->fields()) {
        parts.push_back(unpackValue(field->getType(), words, cursor, loc));
    }
    return requireStruct(record, loc) + "(" + join(parts, ", ") + ")";
}

// Appends the uint expressions of the words of `value`, in packed order.
void Writer::packWords(clang::QualType const original, std::string const &value, std::vector<std::string> &words,
                       clang::SourceLocation const loc) {
    if (auto const encoded = encodedInfo(original, context, diags, loc)) {
        auto const encode = requireFunction(encoded->encode, FunctionKind::Free, loc);
        for (std::uint32_t i = 0; i < encoded->words; ++i) {
            words.push_back(encode + "(" + value + ")[" + std::to_string(i) + "]");
        }
        return;
    }
    auto type = bareType(original);
    if (auto const *enumType = type->getAs<clang::EnumType>()) {
        type = bareType(enumType->getDecl()->getIntegerType());
    }
    if (type->isBooleanType()) {
        words.push_back("(" + value + " ? 1u : 0u)");
        return;
    }
    if (type->isRealFloatingType()) {
        if (typeName(type, loc) == "double") {
            words.push_back("unpackDouble2x32(" + value + ").x");
            words.push_back("unpackDouble2x32(" + value + ").y");
            return;
        }
        words.push_back("floatBitsToUint(" + value + ")");
        return;
    }
    if (type->isIntegerType()) {
        words.push_back(typeName(type, loc) == "uint" ? value : "uint(" + value + ")");
        return;
    }
    if (auto const vec = asTcVec(type)) {
        for (unsigned i = 0; i < vec->size; ++i) {
            packWords(vec->element, value + "[" + std::to_string(i) + "]", words, loc);
        }
        return;
    }
    if (auto const matrix = asTcMat(type)) {
        for (unsigned c = 0; c < matrix->columns; ++c) {
            for (unsigned r = 0; r < matrix->rows; ++r) {
                packWords(matrix->element, value + "[" + std::to_string(c) + "][" + std::to_string(r) + "]", words,
                          loc);
            }
        }
        return;
    }
    if (auto const array = asStdArray(type)) {
        for (std::uint64_t i = 0; i < array->size; ++i) {
            packWords(array->element, value + "[" + std::to_string(i) + "]", words, loc);
        }
        return;
    }
    for (auto const *field : type->getAsCXXRecordDecl()->fields()) {
        packWords(field->getType(), value + "." + fieldName(field), words, loc);
    }
}

// ------------------------------------------------------------------- run

std::optional<GlslBody> Writer::run() {
    for (auto const &buffer : kernel.buffers) {
        auto const elementType = bareType(buffer.elementType);
        if (elementType->isBooleanType() or (asTcVec(elementType) and asTcVec(elementType)->element->isBooleanType())) {
            diags.error(buffer.field->getLocation(), "bool has no GPU memory layout; store uint or int instead");
        }
        out.bufferTypes.push_back(typeName(buffer.elementType, buffer.field->getLocation()));
    }
    for (auto const &uniform : kernel.uniforms) {
        out.uniformTypes.push_back(typeName(uniform.type, uniform.field->getLocation()));
    }
    for (auto const &variable : kernel.shared) {
        out.shared.push_back(declaration(variable.type, variable.name, variable.field->getLocation()));
    }
    requireFunction(kernel.entry, FunctionKind::KernelEntry, kernel.entry->getLocation());
    while (not pending.empty()) {
        auto const index = pending.front();
        pending.pop_front();
        auto text = definition(functions[index]);
        functions[index].definition = std::move(text);
    }
    auto const acyclic = checkRecursion();
    for (std::size_t i = 1; i < functions.size(); ++i) {
        out.prototypes.push_back(signature(functions[i]) + ";");
    }
    for (std::size_t i = 1; i < functions.size(); ++i) {
        out.functions.push_back(functions[i].definition);
    }
    auto entry = functions.front().definition;
    if (not constantInitializers.empty()) {
        auto init = std::string{"void tc_initConstants() {\n"};
        for (auto const &line : constantInitializers) {
            init += "    " + line + "\n";
        }
        out.prototypes.push_back("void tc_initConstants();");
        out.functions.push_back(init + "}");
        entry.insert(entry.find("{\n") + 2, "    tc_initConstants();\n");
    }
    out.functions.push_back(entry);
    if (diags.hasErrors() or not acyclic) {
        return std::nullopt;
    }
    return out;
}

} // namespace

std::optional<GlslBody> writeGlsl(KernelModel &kernel, clang::ASTContext &context, Diagnostics &diags) {
    return Writer{kernel, context, diags}.run();
}

} // namespace tc::transpiler
