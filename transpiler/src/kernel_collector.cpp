#include "kernel_collector.hpp"

#include <algorithm>
#include <map>
#include <string>

#include <clang/AST/Attr.h>
#include <clang/AST/Expr.h>
#include <clang/ASTMatchers/ASTMatchFinder.h>
#include <clang/ASTMatchers/ASTMatchers.h>

#include "glsl_names.hpp"
#include "tc_types.hpp"

namespace tc::transpiler {

namespace {

using namespace clang::ast_matchers;

bool hasKernelAnnotation(clang::Decl const *decl) {
    for (auto const *attr : decl->specific_attrs<clang::AnnotateAttr>()) {
        if (attr->getAnnotation() == "kernel") {
            return true;
        }
    }
    return false;
}

struct KernelMatchCollector final : MatchFinder::MatchCallback {
    std::vector<clang::CXXRecordDecl const *> kernels;

    void run(MatchFinder::MatchResult const &result) override {
        auto const *record = result.Nodes.getNodeAs<clang::CXXRecordDecl>("kernel");
        if (record != nullptr and hasKernelAnnotation(record)) {
            kernels.push_back(record);
        }
    }
};

std::optional<std::array<std::uint32_t, 3>> evaluateLocalSize(clang::FieldDecl const *field,
                                                              clang::ASTContext &context, Diagnostics &diags) {
    auto const vec = asTcVec(field->getType());
    if (not vec or vec->size != 3 or not bareType(vec->element)->isSpecificBuiltinType(clang::BuiltinType::UInt)) {
        diags.error(field->getLocation(), "local_size must be declared as tc::uvec3");
        return std::nullopt;
    }
    auto const *init = field->getInClassInitializer();
    auto result = clang::Expr::EvalResult{};
    if (init == nullptr or not init->EvaluateAsRValue(result, context, true) or not result.Val.isStruct()
        or result.Val.getStructNumBases() != 1) {
        diags.error(field->getLocation(), "local_size needs a constant initializer such as {256, 1, 1}");
        return std::nullopt;
    }
    auto const &fields = result.Val.getStructBase(0);
    auto size = std::array<std::uint32_t, 3>{};
    for (unsigned i = 0; i < 3; ++i) {
        size[i] = static_cast<std::uint32_t>(fields.getStructField(i).getInt().getZExtValue());
        if (size[i] == 0) {
            diags.error(field->getLocation(), "local_size components must be positive");
            return std::nullopt;
        }
    }
    return size;
}

std::optional<std::string> programNameOf(clang::CXXRecordDecl const *record, Diagnostics &diags) {
    for (auto const *decl : record->decls()) {
        auto const *var = llvm::dyn_cast<clang::VarDecl>(decl);
        if (var == nullptr or var->getName() != "fileLocation") {
            continue;
        }
        auto const *init = var->getInit();
        auto const *literal =
            init != nullptr ? llvm::dyn_cast<clang::StringLiteral>(init->IgnoreParenImpCasts()) : nullptr;
        if (not var->isConstexpr() or literal == nullptr or literal->getString().empty()) {
            diags.error(var->getLocation(), "fileLocation must be 'static constexpr char fileLocation[] = \"name\"'");
            return std::nullopt;
        }
        return literal->getString().str();
    }
    diags.error(record->getLocation(), "kernel struct needs 'static constexpr char fileLocation[] = \"name\"'");
    return std::nullopt;
}

struct UniformType {
    tc::ScalarKind scalar;
    std::uint32_t components;
    std::uint32_t columns;
};

std::optional<tc::ScalarKind> scalarKindOf(clang::QualType const type) {
    auto const *builtin = bareType(type)->getAs<clang::BuiltinType>();
    if (builtin == nullptr) {
        return std::nullopt;
    }
    switch (builtin->getKind()) {
    case clang::BuiltinType::Int:
        return tc::ScalarKind::Int;
    case clang::BuiltinType::UInt:
        return tc::ScalarKind::UInt;
    case clang::BuiltinType::Float:
        return tc::ScalarKind::Float;
    case clang::BuiltinType::Double:
        return tc::ScalarKind::Double;
    default:
        return std::nullopt;
    }
}

std::optional<UniformType> uniformTypeOf(clang::QualType const type) {
    if (auto const scalar = scalarKindOf(type)) {
        return UniformType{*scalar, 1, 1};
    }
    if (auto const vec = asTcVec(type)) {
        if (auto const scalar = scalarKindOf(vec->element)) {
            return UniformType{*scalar, vec->size, 1};
        }
    }
    if (auto const mat = asTcMat(type)) {
        if (auto const scalar = scalarKindOf(mat->element); scalar == tc::ScalarKind::Float) {
            return UniformType{*scalar, mat->rows, mat->columns};
        }
    }
    return std::nullopt;
}

// std430 placement of a scalar, vector or matrix inside a push-constant
// block; a matrix is an array of column vectors.
std::uint32_t pushConstantAlignment(UniformModel const &uniform) {
    auto const scalarSize = uniform.scalar == tc::ScalarKind::Double ? 8u : 4u;
    if (uniform.components == 1) {
        return scalarSize;
    }
    return (uniform.components == 2 ? 2u : 4u) * scalarSize;
}

std::uint32_t pushConstantSize(UniformModel const &uniform) {
    auto const scalarSize = uniform.scalar == tc::ScalarKind::Double ? 8u : 4u;
    if (uniform.columns == 1) {
        return uniform.components * scalarSize;
    }
    auto const align = pushConstantAlignment(uniform);
    auto const stride = (uniform.components * scalarSize + align - 1) / align * align;
    return uniform.columns * stride;
}

void layoutPushConstants(KernelModel &model) {
    std::sort(model.uniforms.begin(), model.uniforms.end(),
              [](UniformModel const &a, UniformModel const &b) { return a.location < b.location; });
    auto offset = std::uint32_t{0};
    for (auto &uniform : model.uniforms) {
        auto const align = pushConstantAlignment(uniform);
        offset = (offset + align - 1) / align * align;
        uniform.pushConstantOffset = offset;
        offset += pushConstantSize(uniform);
    }
    model.pushConstantSize = (offset + 3) / 4 * 4;
}

std::optional<tc::ImageFormat> imageFormatOf(clang::QualType const formatType, clang::ASTContext &context) {
    auto const *spec = asTcSpecialization(formatType, "gpu::Format");
    if (spec == nullptr) {
        return std::nullopt;
    }
    auto const storage = templateTypeArg(spec, 1);
    auto format = tc::ImageFormat{};
    format.encoding = static_cast<tc::Encoding>(templateIntArg(spec, 0));
    format.isSigned = bareType(storage)->isSignedIntegerType();
    format.bitsPerChannel = static_cast<std::uint8_t>(context.getTypeSize(storage));
    format.channelCount = static_cast<std::uint8_t>(templateIntArg(spec, 2));
    return format;
}

bool classifyField(clang::FieldDecl const *field, KernelModel &model, clang::ASTContext &context,
                   Diagnostics &diags) {
    auto const name = glslIdentifier(field->getName());
    if (auto const *spec = asTcSpecialization(field->getType(), "BufferBinding")) {
        auto buffer = BufferModel{};
        buffer.name = name;
        buffer.elementType = templateTypeArg(spec, 0);
        buffer.binding = static_cast<std::uint32_t>(templateIntArg(spec, 1));
        buffer.field = field;
        model.buffers.push_back(buffer);
        return true;
    }
    if (auto const *spec = asTcSpecialization(field->getType(), "ImageBinding")) {
        auto const format = imageFormatOf(templateTypeArg(spec, 0), context);
        if (not format) {
            diags.error(field->getLocation(), "the first ImageBinding argument must be a tc::gpu format");
            return false;
        }
        auto image = ImageModel{};
        image.name = name;
        image.format = *format;
        image.dim = static_cast<tc::Dim>(templateIntArg(spec, 1));
        image.binding = static_cast<std::uint32_t>(templateIntArg(spec, 3));
        image.field = field;
        model.images.push_back(image);
        return true;
    }
    if (auto const *spec = asTcSpecialization(field->getType(), "TextureBinding")) {
        auto const format = imageFormatOf(templateTypeArg(spec, 0), context);
        if (not format) {
            diags.error(field->getLocation(), "the first TextureBinding argument must be a tc::gpu format");
            return false;
        }
        auto texture = TextureModel{};
        texture.name = name;
        texture.format = *format;
        texture.dim = static_cast<tc::Dim>(templateIntArg(spec, 1));
        texture.binding = static_cast<std::uint32_t>(templateIntArg(spec, 3));
        texture.filter = static_cast<tc::Filter>(templateIntArg(spec, 4));
        texture.wrap = static_cast<tc::Wrap>(templateIntArg(spec, 5));
        texture.field = field;
        model.textures.push_back(texture);
        return true;
    }
    if (auto const *spec = asTcSpecialization(field->getType(), "Shared")) {
        auto variable = SharedModel{};
        variable.name = name;
        variable.type = templateTypeArg(spec, 0);
        variable.field = field;
        model.shared.push_back(variable);
        return true;
    }
    if (auto const *spec = asTcSpecialization(field->getType(), "ExternalTextureBinding")) {
        auto texture = ExternalTextureModel{};
        texture.name = name;
        texture.binding = static_cast<std::uint32_t>(templateIntArg(spec, 0));
        texture.field = field;
        model.externalTextures.push_back(texture);
        return true;
    }
    if (auto const *spec = asTcSpecialization(field->getType(), "Uniform")) {
        auto const valueType = templateTypeArg(spec, 0);
        auto const type = uniformTypeOf(valueType);
        if (not type) {
            diags.error(field->getLocation(),
                        "Uniform<T> supports int, uint, float, double, their vectors and float matrices");
            return false;
        }
        auto uniform = UniformModel{};
        uniform.name = name;
        uniform.location = static_cast<std::uint32_t>(templateIntArg(spec, 1));
        uniform.type = valueType;
        uniform.scalar = type->scalar;
        uniform.components = type->components;
        uniform.columns = type->columns;
        uniform.field = field;
        model.uniforms.push_back(uniform);
        return true;
    }
    if (field->getName() == "local_size") {
        auto const size = evaluateLocalSize(field, context, diags);
        if (not size) {
            return false;
        }
        model.localSize = *size;
    }
    return true;
}

bool checkUniqueSlots(KernelModel const &model, Diagnostics &diags) {
    auto ok = true;
    auto bindings = std::map<std::uint32_t, clang::FieldDecl const *>{};
    auto const claim = [&](std::uint32_t const binding, clang::FieldDecl const *field) {
        auto const [it, inserted] = bindings.emplace(binding, field);
        if (not inserted) {
            diags.error(field->getLocation(), "binding " + std::to_string(binding) + " is already used by '"
                                                  + it->second->getName().str()
                                                  + "'; buffers and images share one Vulkan descriptor set");
            ok = false;
        }
    };
    for (auto const &buffer : model.buffers) {
        claim(buffer.binding, buffer.field);
    }
    for (auto const &image : model.images) {
        claim(image.binding, image.field);
    }
    for (auto const &texture : model.textures) {
        claim(texture.binding, texture.field);
    }
    for (auto const &texture : model.externalTextures) {
        claim(texture.binding, texture.field);
    }
    auto locations = std::map<std::uint32_t, clang::FieldDecl const *>{};
    for (auto const &uniform : model.uniforms) {
        if (not locations.emplace(uniform.location, uniform.field).second) {
            diags.error(uniform.field->getLocation(),
                        "uniform location " + std::to_string(uniform.location) + " is used twice");
            ok = false;
        }
    }
    return ok;
}

clang::CXXMethodDecl const *findEntry(clang::CXXRecordDecl const *record, Diagnostics &diags) {
    for (auto const *method : record->methods()) {
        if (method->getDeclName().isIdentifier() and method->getName() == "main") {
            if (method->isStatic() or method->getNumParams() != 0 or not method->getReturnType()->isVoidType()
                or not method->hasBody()) {
                diags.error(method->getLocation(), "kernel entry point must be 'void main()' with a body");
                return nullptr;
            }
            return method;
        }
    }
    diags.error(record->getLocation(), "kernel struct needs a 'void main()' member function");
    return nullptr;
}

} // namespace

std::vector<clang::CXXRecordDecl const *> findKernels(clang::ASTContext &context, Diagnostics &diags) {
    auto collector = KernelMatchCollector{};
    auto finder = MatchFinder{};
    finder.addMatcher(cxxRecordDecl(isDefinition(), isExpansionInMainFile(), hasAttr(clang::attr::Annotate),
                                    unless(isTemplateInstantiation()))
                          .bind("kernel"),
                      &collector);
    finder.matchAST(context);
    auto kernels = std::vector<clang::CXXRecordDecl const *>{};
    for (auto const *record : collector.kernels) {
        if (record->getDescribedClassTemplate() != nullptr) {
            diags.error(record->getLocation(), "kernel structs cannot be templates");
            continue;
        }
        kernels.push_back(record);
    }
    return kernels;
}

std::optional<KernelModel> collectKernel(clang::CXXRecordDecl const *record, clang::ASTContext &context,
                                         Diagnostics &diags) {
    auto model = KernelModel{};
    model.record = record;
    auto const name = programNameOf(record, diags);
    model.entry = findEntry(record, diags);
    auto ok = name.has_value() and model.entry != nullptr;
    if (name) {
        model.programName = *name;
    }
    auto hasLocalSize = false;
    for (auto const *field : record->fields()) {
        ok = classifyField(field, model, context, diags) and ok;
        hasLocalSize = hasLocalSize or field->getName() == "local_size";
    }
    if (not hasLocalSize) {
        diags.error(record->getLocation(), "kernel struct needs 'tc::uvec3 local_size{x, y, z}'");
        ok = false;
    }
    if (record->getNumBases() != 0) {
        diags.error(record->getLocation(), "kernel structs cannot have base classes");
        ok = false;
    }
    ok = checkUniqueSlots(model, diags) and ok;
    if (not ok) {
        return std::nullopt;
    }
    for (auto const *decl : record->decls()) {
        auto const *var = llvm::dyn_cast<clang::VarDecl>(decl);
        if (var != nullptr and var->getName() == "usesBarrier") {
            auto const *value = var->isConstexpr() ? var->evaluateValue() : nullptr;
            model.usesBarrier = value != nullptr and value->isInt() and value->getInt().getBoolValue();
        }
    }
    auto sharedBytes = std::uint64_t{0};
    for (auto const &variable : model.shared) {
        sharedBytes += static_cast<std::uint64_t>(context.getTypeSizeInChars(variable.type).getQuantity());
    }
    if (sharedBytes > 16384) {
        diags.warning(record->getLocation(), "kernel uses " + std::to_string(sharedBytes)
                                                 + " bytes of shared memory; OpenGL ES and Vulkan only guarantee 16384");
    }
    layoutPushConstants(model);
    return model;
}

} // namespace tc::transpiler
