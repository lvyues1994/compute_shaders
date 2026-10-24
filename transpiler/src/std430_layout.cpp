#include "std430_layout.hpp"

#include <algorithm>
#include <optional>
#include <string>

#include <clang/AST/RecordLayout.h>

#include "tc_types.hpp"

namespace tc::transpiler {

namespace {

struct Std430 {
    std::uint64_t size = 0;
    std::uint64_t align = 1;
};

std::uint64_t roundUp(std::uint64_t const value, std::uint64_t const align) {
    return (value + align - 1) / align * align;
}

struct LayoutChecker {
    clang::ASTContext &context;
    Diagnostics &diags;
    clang::SourceLocation loc;

    std::uint64_t cppSize(clang::QualType const type) const {
        return static_cast<std::uint64_t>(context.getTypeSizeInChars(type).getQuantity());
    }

    std::optional<Std430> scalar(clang::QualType const type) {
        if (type->isBooleanType()) {
            diags.error(loc, "bool has no std430 layout; use uint or int in buffers");
            return std::nullopt;
        }
        auto const size = cppSize(type);
        if ((size != 4 and size != 8) or not(type->isIntegerType() or type->isRealFloatingType())) {
            diags.error(loc, "'" + type.getAsString() + "' cannot be stored in a GLSL buffer");
            return std::nullopt;
        }
        return Std430{size, size};
    }

    std::optional<Std430> array(clang::QualType const element, std::uint64_t const count) {
        auto const inner = of(element);
        if (not inner) {
            return std::nullopt;
        }
        auto const stride = roundUp(inner->size, inner->align);
        if (stride != cppSize(element)) {
            diags.error(loc, "array of '" + element.getAsString() + "' has a " + std::to_string(cppSize(element))
                                 + "-byte stride in C++ but " + std::to_string(stride) + " bytes in std430");
            return std::nullopt;
        }
        return Std430{stride * count, inner->align};
    }

    std::optional<Std430> record(clang::CXXRecordDecl const *decl) {
        auto const &layout = context.getASTRecordLayout(decl);
        auto offset = std::uint64_t{0};
        auto align = std::uint64_t{1};
        auto index = 0u;
        for (auto const *field : decl->fields()) {
            auto const member = of(field->getType());
            if (not member) {
                return std::nullopt;
            }
            offset = roundUp(offset, member->align);
            auto const cppOffset = layout.getFieldOffset(index++) / 8;
            if (cppOffset != offset) {
                diags.error(field->getLocation(),
                            "std430 layout mismatch in '" + decl->getName().str() + "': member '"
                                + field->getName().str() + "' is at byte " + std::to_string(cppOffset) + " in C++ but "
                                + std::to_string(offset) + " in GLSL; reorder members or use vec4");
                return std::nullopt;
            }
            offset += member->size;
            align = std::max(align, member->align);
        }
        auto const size = roundUp(offset, align);
        if (size != cppSize(context.getRecordType(decl))) {
            diags.error(decl->getLocation(), "std430 size of '" + decl->getName().str() + "' is "
                                                 + std::to_string(size) + " bytes but sizeof is "
                                                 + std::to_string(cppSize(context.getRecordType(decl))));
            return std::nullopt;
        }
        return Std430{size, align};
    }

    std::optional<Std430> of(clang::QualType const original) {
        auto const type = bareType(original);
        if (auto const vec = asTcVec(type)) {
            auto const component = scalar(bareType(vec->element));
            if (not component) {
                return std::nullopt;
            }
            auto const align = (vec->size == 2 ? 2 : 4) * component->size;
            return Std430{vec->size * component->size, align};
        }
        if (auto const matrix = asTcMat(type)) {
            // Column-major: an array of `columns` vectors of `rows` components.
            auto const component = scalar(bareType(matrix->element));
            if (not component) {
                return std::nullopt;
            }
            auto const align = (matrix->rows == 2 ? 2 : 4) * component->size;
            auto const stride = roundUp(matrix->rows * component->size, align);
            return Std430{matrix->columns * stride, align};
        }
        if (auto const arr = asStdArray(type)) {
            return array(arr->element, arr->size);
        }
        if (auto const *carray = context.getAsConstantArrayType(type)) {
            return array(carray->getElementType(), carray->getSize().getZExtValue());
        }
        if (auto const *decl = type->getAsCXXRecordDecl()) {
            return record(decl);
        }
        return scalar(type);
    }
};

} // namespace

bool checkBufferLayouts(KernelModel const &kernel, clang::ASTContext &context, Diagnostics &diags) {
    auto ok = true;
    for (auto const &buffer : kernel.buffers) {
        auto checker = LayoutChecker{context, diags, buffer.field->getLocation()};
        auto const layout = checker.of(buffer.elementType);
        if (not layout) {
            ok = false;
            continue;
        }
        auto const stride = roundUp(layout->size, layout->align);
        auto const cppStride = checker.cppSize(buffer.elementType);
        if (stride != cppStride) {
            diags.error(buffer.field->getLocation(),
                        "buffer '" + buffer.name + "' has a " + std::to_string(cppStride)
                            + "-byte element stride in C++ but " + std::to_string(stride) + " bytes in std430");
            ok = false;
        }
    }
    return ok;
}

} // namespace tc::transpiler
