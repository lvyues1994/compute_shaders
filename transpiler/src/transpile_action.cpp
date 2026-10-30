#include "transpile_action.hpp"

#include <clang/AST/ASTConsumer.h>
#include <clang/AST/ASTContext.h>
#include <clang/Basic/SourceManager.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>

#include "diagnostics.hpp"
#include "glsl_writer.hpp"
#include "kernel_collector.hpp"
#include "shader_assembler.hpp"
#include "std430_layout.hpp"

namespace tc::transpiler {

namespace {

KernelArtifact makeArtifact(KernelModel const &model, ShaderVariants const &shaders) {
    auto artifact = KernelArtifact{};
    artifact.name = model.programName;
    artifact.localSize = model.localSize;
    artifact.glslDesktop = shaders.desktop;
    artifact.glslEs = shaders.es;
    artifact.glslVulkan = shaders.vulkan;
    artifact.esUnavailableReason = shaders.esUnavailableReason;
    artifact.pushConstantSize = model.pushConstantSize;
    for (auto const &buffer : model.buffers) {
        auto resource = ResourceArtifact{};
        resource.name = buffer.name;
        resource.binding = buffer.binding;
        resource.kind = tc::ResourceKind::StorageBuffer;
        resource.access = accessOf(buffer.usage);
        artifact.resources.push_back(resource);
    }
    for (auto const &table : model.tables) {
        auto resource = ResourceArtifact{};
        resource.name = table.name;
        resource.binding = table.binding;
        resource.kind = tc::ResourceKind::StorageBuffer;
        resource.access = accessOf(table.usage);
        artifact.resources.push_back(resource);
        auto check = TableCheck{};
        check.elementType = table.elementCppName;
        check.words = table.layout.words;
        if (table.layout.isRecord) {
            for (auto const &field : table.layout.fields) {
                check.fieldOffsets.push_back(field.offset);
            }
        }
        artifact.tableChecks.push_back(check);
    }
    for (auto const &image : model.images) {
        auto resource = ResourceArtifact{};
        resource.name = image.name;
        resource.binding = image.binding;
        resource.kind = tc::ResourceKind::StorageImage;
        auto const used = image.usage.read or image.usage.written;
        resource.access = used ? accessOf(image.usage) : tc::Access::ReadOnly;
        resource.dim = image.dim;
        resource.format = image.format;
        artifact.resources.push_back(resource);
    }
    for (auto const &texture : model.textures) {
        auto resource = ResourceArtifact{};
        resource.name = texture.name;
        resource.binding = texture.binding;
        resource.kind = tc::ResourceKind::SampledTexture;
        resource.access = tc::Access::ReadOnly;
        resource.dim = texture.dim;
        resource.format = texture.format;
        resource.filter = texture.filter;
        resource.wrap = texture.wrap;
        artifact.resources.push_back(resource);
    }
    for (auto const &texture : model.externalTextures) {
        auto resource = ResourceArtifact{};
        resource.name = texture.name;
        resource.binding = texture.binding;
        resource.kind = tc::ResourceKind::ExternalTexture;
        resource.access = tc::Access::ReadOnly;
        resource.filter = tc::Filter::Linear;
        artifact.resources.push_back(resource);
    }
    for (auto const &uniform : model.uniforms) {
        auto slot = UniformArtifact{};
        slot.name = uniform.name;
        slot.location = uniform.location;
        slot.scalar = uniform.scalar;
        slot.components = uniform.components;
        slot.columns = uniform.columns;
        slot.pushConstantOffset = uniform.pushConstantOffset;
        artifact.uniforms.push_back(slot);
    }
    return artifact;
}

struct TranspileConsumer final : clang::ASTConsumer {
    TranspileConsumer(TranspileResult &result_, std::string sourceName_)
        : result(result_), sourceName(std::move(sourceName_)) {}

    void HandleTranslationUnit(clang::ASTContext &context) override {
        if (context.getDiagnostics().hasErrorOccurred()) {
            return;
        }
        auto diags = Diagnostics{context.getDiagnostics()};
        auto const &sources = context.getSourceManager();
        auto const kernels = findKernels(context, diags);
        if (kernels.empty()) {
            diags.warning(sources.getLocForStartOfFile(sources.getMainFileID()),
                          "no struct marked [[clang::annotate(\"kernel\")]] (TC_KERNEL) in this file");
        }
        for (auto const *record : kernels) {
            transpile(record, context, diags);
        }
        for (auto it = sources.fileinfo_begin(); it != sources.fileinfo_end(); ++it) {
            result.dependencies.push_back(it->first.getName().str());
        }
        result.ok = not diags.hasErrors();
    }

private:
    void transpile(clang::CXXRecordDecl const *record, clang::ASTContext &context, Diagnostics &diags) {
        auto model = collectKernel(record, context, diags);
        if (not model) {
            return;
        }
        auto const body = writeGlsl(*model, context, diags);
        if (not body or not checkBufferLayouts(*model, context, diags)) {
            return;
        }
        auto const shaders = assembleShaders(*model, *body, sourceName);
        result.kernels.push_back(makeArtifact(*model, shaders));
    }

    TranspileResult &result;
    std::string sourceName;
};

struct TranspileAction final : clang::ASTFrontendAction {
    TranspileAction(TranspileResult &result_, std::string sourceName_)
        : result(result_), sourceName(std::move(sourceName_)) {}

    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance &, llvm::StringRef) override {
        return std::make_unique<TranspileConsumer>(result, sourceName);
    }

private:
    TranspileResult &result;
    std::string sourceName;
};

struct TranspileActionFactory final : clang::tooling::FrontendActionFactory {
    TranspileActionFactory(TranspileResult &result_, std::string sourceName_)
        : result(result_), sourceName(std::move(sourceName_)) {}

    std::unique_ptr<clang::FrontendAction> create() override {
        return std::make_unique<TranspileAction>(result, sourceName);
    }

private:
    TranspileResult &result;
    std::string sourceName;
};

} // namespace

std::unique_ptr<clang::tooling::FrontendActionFactory> makeTranspileActionFactory(TranspileResult &result,
                                                                                  std::string sourceName) {
    return std::make_unique<TranspileActionFactory>(result, std::move(sourceName));
}

} // namespace tc::transpiler
