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

// One field of a table element in the packed word layout (tc/table.hpp).
struct PackedField {
    clang::FieldDecl const *decl = nullptr; // nullptr: the element itself (vector, matrix or array tables)
    clang::QualType type;
    std::uint32_t offset = 0; // first word within the packed record
    std::uint32_t words = 0;
};

struct PackedLayout {
    std::uint32_t words = 0;
    bool isRecord = false;
    std::vector<PackedField> fields;
};

// A table layout type (tc::Aos, tc::Soa<P> or a user layout, see
// tc/table.hpp): fieldAddress is translated to GLSL, fieldAlignment evaluated.
struct TableAddressing {
    clang::CXXRecordDecl const *record = nullptr;
    clang::CXXMethodDecl const *fieldAddress = nullptr;
    clang::CXXMethodDecl const *fieldAlignment = nullptr;
    bool contiguousRecords = true;
};

struct TableModel {
    std::string name;
    std::uint32_t binding = 0;
    clang::QualType elementType;
    std::string elementCppName; // fully qualified, for the generated layout checks
    TableAddressing addressing;
    PackedLayout layout;
    clang::FieldDecl const *field = nullptr;
    AccessUsage usage;
    // uvec2 / uvec4 views of the same buffer used by the generated accessors.
    bool usesView2 = false;
    bool usesView4 = false;
};

struct ImageModel {
    std::string name;
    std::uint32_t binding = 0;
    tc::ImageFormat format;
    tc::Dim dim = tc::Dim::D2;
    bool external = false; // ExternalImageBinding: a platform frame
    clang::FieldDecl const *field = nullptr;
    AccessUsage usage;
};

struct TextureModel {
    std::string name;
    std::uint32_t binding = 0;
    tc::ImageFormat format;
    tc::Dim dim = tc::Dim::D2;
    tc::Filter filter = tc::Filter::Nearest;
    tc::Wrap wrap = tc::Wrap::ClampToEdge;
    tc::Mip mip = tc::Mip::None;
    clang::FieldDecl const *field = nullptr;
};

struct ExternalTextureModel {
    std::string name;
    std::uint32_t binding = 0;
    clang::FieldDecl const *field = nullptr;
};

struct SharedModel {
    std::string name;
    clang::QualType type;
    clang::FieldDecl const *field = nullptr;
};

struct UniformModel {
    std::string name;
    std::uint32_t location = 0;
    clang::QualType type;
    tc::ScalarKind scalar = tc::ScalarKind::Float;
    std::uint32_t components = 1;
    std::uint32_t columns = 1;
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
    std::vector<TableModel> tables;
    std::vector<ImageModel> images;
    std::vector<TextureModel> textures;
    std::vector<ExternalTextureModel> externalTextures;
    std::vector<SharedModel> shared;
    std::vector<UniformModel> uniforms;
    std::uint32_t pushConstantSize = 0;
    std::uint32_t sharedBytes = 0;
    bool usesBarrier = false;   // `static constexpr bool usesBarrier = true;`
    bool usesSubgroups = false; // `static constexpr bool usesSubgroups = true;`
    bool mediumFloats = false;  // `static constexpr tc::Precision floatPrecision = tc::Precision::Medium;`

    bool isShared(clang::FieldDecl const *field) const {
        for (auto const &variable : shared) {
            if (variable.field == field) {
                return true;
            }
        }
        return false;
    }

    bool isTexture(clang::FieldDecl const *field) const {
        for (auto const &texture : textures) {
            if (texture.field == field) {
                return true;
            }
        }
        for (auto const &texture : externalTextures) {
            if (texture.field == field) {
                return true;
            }
        }
        return false;
    }

    BufferModel *findBuffer(clang::FieldDecl const *field) {
        for (auto &buffer : buffers) {
            if (buffer.field == field) {
                return &buffer;
            }
        }
        return nullptr;
    }

    TableModel *findTable(clang::FieldDecl const *field) {
        for (auto &table : tables) {
            if (table.field == field) {
                return &table;
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

// GL_KHR_shader_subgroup_* extensions a kernel needs (bit flags).
enum SubgroupExtension : std::uint32_t {
    kSubgroupBasic = 1u << 0,
    kSubgroupVote = 1u << 1,
    kSubgroupArithmetic = 1u << 2,
    kSubgroupBallot = 1u << 3,
    kSubgroupShuffle = 1u << 4,
    kSubgroupShuffleRelative = 1u << 5,
};

// GLSL shared by every dialect: everything after the resource declarations.
struct GlslBody {
    std::vector<std::string> bufferTypes;  // GLSL element type per KernelModel::buffers entry
    std::vector<std::string> uniformTypes; // GLSL type per KernelModel::uniforms entry
    std::vector<std::string> shared;       // `T name[N]` per KernelModel::shared entry
    std::vector<std::string> structs;
    std::vector<std::string> constants;
    std::vector<std::string> tableAccessors; // generated get/set functions of the tables
    std::vector<std::string> prototypes;
    std::vector<std::string> functions;
    bool usesDouble = false;
    std::uint32_t features = 0;           // tc::KernelFeature bits
    std::uint32_t subgroupExtensions = 0; // SubgroupExtension bits
};

// AST-independent result for one kernel, consumed by the output writers.
struct ResourceArtifact {
    std::string name;
    std::uint32_t binding = 0;
    tc::ResourceKind kind = tc::ResourceKind::StorageBuffer;
    tc::Access access = tc::Access::ReadWrite;
    tc::Dim dim = tc::Dim::D2;
    tc::ImageFormat format;
    tc::Filter filter = tc::Filter::Nearest;
    tc::Wrap wrap = tc::Wrap::ClampToEdge;
    tc::Mip mip = tc::Mip::None;
};

struct UniformArtifact {
    std::string name;
    std::uint32_t location = 0;
    tc::ScalarKind scalar = tc::ScalarKind::Float;
    std::uint32_t components = 1;
    std::uint32_t columns = 1;
    std::uint32_t pushConstantOffset = 0;
};

// The packed layout tc-transpile assumed for a table element; the generated
// source asserts that tc/table.hpp computes the same.
struct TableCheck {
    std::string elementType;
    std::uint32_t words = 0;
    std::vector<std::uint32_t> fieldOffsets; // empty unless the element is a struct
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
    std::vector<TableCheck> tableChecks;
    std::uint32_t pushConstantSize = 0;
    std::uint32_t sharedBytes = 0;
    std::uint32_t features = 0;
};

} // namespace tc::transpiler
