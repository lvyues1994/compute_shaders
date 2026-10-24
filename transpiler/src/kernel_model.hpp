#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <clang/AST/Decl.h>
#include <clang/AST/DeclCXX.h>
#include <clang/AST/Type.h>

#include <tc/program.hpp>

namespace tc::transpiler {

struct AccessUsage {
    bool read = false;
    bool written = false;
};

inline tc::Access accessOf(AccessUsage const usage) noexcept {
    if (usage.read and not usage.written) {
        return tc::Access::ReadOnly;
    }
    if (usage.written and not usage.read) {
        return tc::Access::WriteOnly;
    }
    return tc::Access::ReadWrite;
}

struct BufferModel {
    std::string name;
    std::uint32_t binding = 0;
    clang::QualType elementType;
    clang::FieldDecl const *field = nullptr;
    AccessUsage usage;
};

struct ImageModel {
    std::string name;
    std::uint32_t binding = 0;
    tc::ImageFormat format;
    tc::Dim dim = tc::Dim::D2;
    clang::FieldDecl const *field = nullptr;
    AccessUsage usage;
};

struct UniformModel {
    std::string name;
    std::uint32_t location = 0;
    clang::QualType type;
    tc::ScalarKind scalar = tc::ScalarKind::Float;
    std::uint32_t components = 1;
    std::uint32_t pushConstantOffset = 0;
    clang::FieldDecl const *field = nullptr;
};

// What the transpiler knows about one kernel struct while its AST is alive.
struct KernelModel {
    clang::CXXRecordDecl const *record = nullptr;
    clang::CXXMethodDecl const *entry = nullptr;
    std::string programName;
    std::array<std::uint32_t, 3> localSize{1, 1, 1};
    std::vector<BufferModel> buffers;
    std::vector<ImageModel> images;
    std::vector<UniformModel> uniforms;
    std::uint32_t pushConstantSize = 0;

    BufferModel *findBuffer(clang::FieldDecl const *field) {
        for (auto &buffer : buffers) {
            if (buffer.field == field) {
                return &buffer;
            }
        }
        return nullptr;
    }

    ImageModel *findImage(clang::FieldDecl const *field) {
        for (auto &image : images) {
            if (image.field == field) {
                return &image;
            }
        }
        return nullptr;
    }

    UniformModel const *findUniform(clang::FieldDecl const *field) const {
        for (auto const &uniform : uniforms) {
            if (uniform.field == field) {
                return &uniform;
            }
        }
        return nullptr;
    }
};

// GLSL shared by every dialect: everything after the resource declarations.
struct GlslBody {
    std::vector<std::string> bufferTypes;  // GLSL element type per KernelModel::buffers entry
    std::vector<std::string> uniformTypes; // GLSL type per KernelModel::uniforms entry
    std::vector<std::string> structs;
    std::vector<std::string> constants;
    std::vector<std::string> prototypes;
    std::vector<std::string> functions;
    bool usesDouble = false;
};

// AST-independent result for one kernel, consumed by the output writers.
struct ResourceArtifact {
    std::string name;
    std::uint32_t binding = 0;
    tc::ResourceKind kind = tc::ResourceKind::StorageBuffer;
    tc::Access access = tc::Access::ReadWrite;
    tc::Dim dim = tc::Dim::D2;
    tc::ImageFormat format;
};

struct UniformArtifact {
    std::string name;
    std::uint32_t location = 0;
    tc::ScalarKind scalar = tc::ScalarKind::Float;
    std::uint32_t components = 1;
    std::uint32_t pushConstantOffset = 0;
};

struct KernelArtifact {
    std::string name;
    std::array<std::uint32_t, 3> localSize{1, 1, 1};
    std::string glslDesktop;
    std::string glslEs;
    std::string glslVulkan;
    std::string esUnavailableReason;
    std::vector<std::uint32_t> spirv;
    std::vector<ResourceArtifact> resources;
    std::vector<UniformArtifact> uniforms;
    std::uint32_t pushConstantSize = 0;
};

} // namespace tc::transpiler
