#include "registration_writer.hpp"

#include <cstdio>

namespace tc::transpiler {

namespace {

char const *accessName(tc::Access const access) {
    switch (access) {
    case tc::Access::ReadOnly:
        return "tc::Access::ReadOnly";
    case tc::Access::WriteOnly:
        return "tc::Access::WriteOnly";
    case tc::Access::ReadWrite:
        return "tc::Access::ReadWrite";
    }
    return "tc::Access::ReadWrite";
}

char const *encodingName(tc::Encoding const encoding) {
    switch (encoding) {
    case tc::Encoding::Normalized:
        return "tc::Encoding::Normalized";
    case tc::Encoding::Float:
        return "tc::Encoding::Float";
    case tc::Encoding::Integer:
        return "tc::Encoding::Integer";
    }
    return "tc::Encoding::Float";
}

char const *scalarName(tc::ScalarKind const scalar) {
    switch (scalar) {
    case tc::ScalarKind::Int:
        return "tc::ScalarKind::Int";
    case tc::ScalarKind::UInt:
        return "tc::ScalarKind::UInt";
    case tc::ScalarKind::Float:
        return "tc::ScalarKind::Float";
    case tc::ScalarKind::Double:
        return "tc::ScalarKind::Double";
    }
    return "tc::ScalarKind::Float";
}

std::string rawString(std::string const &text) {
    return "R\"tc_glsl(" + text + ")tc_glsl\"";
}

std::string spirvArray(std::vector<std::uint32_t> const &words) {
    auto text = std::string{};
    char buffer[16];
    for (std::size_t i = 0; i < words.size(); ++i) {
        std::snprintf(buffer, sizeof buffer, "0x%08xu,", words[i]);
        text += (i % 8 == 0 ? "\n    " : " ") + std::string{buffer};
    }
    return text;
}

std::string resourceSlot(ResourceArtifact const &resource) {
    auto text = std::string{"    [] {\n        auto slot = tc::ResourceSlot{};\n"};
    text += "        slot.name = \"" + resource.name + "\";\n";
    text += "        slot.binding = " + std::to_string(resource.binding) + ";\n";
    text += "        slot.access = " + std::string{accessName(resource.access)} + ";\n";
    if (resource.kind == tc::ResourceKind::ExternalTexture) {
        text += "        slot.kind = tc::ResourceKind::ExternalTexture;\n";
    } else if (resource.kind != tc::ResourceKind::StorageBuffer) {
        auto const isTexture = resource.kind == tc::ResourceKind::SampledTexture;
        text += isTexture ? "        slot.kind = tc::ResourceKind::SampledTexture;\n"
                          : "        slot.kind = tc::ResourceKind::StorageImage;\n";
        if (isTexture) {
            text += std::string{"        slot.filter = "}
                  + (resource.filter == tc::Filter::Linear ? "tc::Filter::Linear" : "tc::Filter::Nearest") + ";\n";
            text += std::string{"        slot.wrap = "}
                  + (resource.wrap == tc::Wrap::Repeat           ? "tc::Wrap::Repeat"
                     : resource.wrap == tc::Wrap::MirroredRepeat ? "tc::Wrap::MirroredRepeat"
                                                                 : "tc::Wrap::ClampToEdge")
                  + ";\n";
        }
        text += "        slot.imageDim = tc::Dim::D" + std::to_string(static_cast<int>(resource.dim)) + ";\n";
        text += "        slot.imageFormat.encoding = " + std::string{encodingName(resource.format.encoding)} + ";\n";
        text += "        slot.imageFormat.isSigned = " + std::string{resource.format.isSigned ? "true" : "false"} + ";\n";
        text += "        slot.imageFormat.bitsPerChannel = " + std::to_string(resource.format.bitsPerChannel) + ";\n";
        text += "        slot.imageFormat.channelCount = " + std::to_string(resource.format.channelCount) + ";\n";
    } else {
        text += "        slot.kind = tc::ResourceKind::StorageBuffer;\n";
    }
    return text + "        return slot;\n    }(),\n";
}

std::string uniformSlot(UniformArtifact const &uniform) {
    auto text = std::string{"    [] {\n        auto slot = tc::UniformSlot{};\n"};
    text += "        slot.name = \"" + uniform.name + "\";\n";
    text += "        slot.location = " + std::to_string(uniform.location) + ";\n";
    text += "        slot.scalar = " + std::string{scalarName(uniform.scalar)} + ";\n";
    text += "        slot.components = " + std::to_string(uniform.components) + ";\n";
    text += "        slot.columns = " + std::to_string(uniform.columns) + ";\n";
    text += "        slot.pushConstantOffset = " + std::to_string(uniform.pushConstantOffset) + ";\n";
    return text + "        return slot;\n    }(),\n";
}

std::string kernelBlock(KernelArtifact const &kernel, std::size_t const index) {
    auto const id = std::to_string(index);
    auto text = "// Kernel \"" + kernel.name + "\"\n";
    text += "constexpr char kGlslDesktop" + id + "[] = " + rawString(kernel.glslDesktop) + ";\n";
    text += "constexpr char kGlslEs" + id + "[] = " + rawString(kernel.glslEs) + ";\n";
    text += "constexpr char kEsUnavailable" + id + "[] = " + rawString(kernel.esUnavailableReason) + ";\n";
    if (not kernel.spirv.empty()) {
        text += "constexpr std::uint32_t kSpirv" + id + "[] = {" + spirvArray(kernel.spirv) + "\n};\n";
    }
    if (not kernel.resources.empty()) {
        text += "tc::ResourceSlot const kResources" + id + "[] = {\n";
        for (auto const &resource : kernel.resources) {
            text += resourceSlot(resource);
        }
        text += "};\n";
    }
    if (not kernel.uniforms.empty()) {
        text += "tc::UniformSlot const kUniforms" + id + "[] = {\n";
        for (auto const &uniform : kernel.uniforms) {
            text += uniformSlot(uniform);
        }
        text += "};\n";
    }
    text += "tc::ProgramRegistration const kRegistration" + id + "{[] {\n";
    text += "    auto program = tc::KernelProgram{};\n";
    text += "    program.name = \"" + kernel.name + "\";\n";
    text += "    program.localSize = tc::uvec3{" + std::to_string(kernel.localSize[0]) + "u, "
          + std::to_string(kernel.localSize[1]) + "u, " + std::to_string(kernel.localSize[2]) + "u};\n";
    text += "    program.glslDesktop = std::string_view{kGlslDesktop" + id + ", sizeof kGlslDesktop" + id + " - 1};\n";
    text += "    program.glslEs = std::string_view{kGlslEs" + id + ", sizeof kGlslEs" + id + " - 1};\n";
    text += "    program.esUnavailableReason = std::string_view{kEsUnavailable" + id + ", sizeof kEsUnavailable" + id
          + " - 1};\n";
    if (not kernel.spirv.empty()) {
        text += "    program.spirv = tc::Span<std::uint32_t const>{kSpirv" + id + ", std::size(kSpirv" + id + ")};\n";
    }
    if (not kernel.resources.empty()) {
        text += "    program.resources = tc::Span<tc::ResourceSlot const>{kResources" + id + ", std::size(kResources"
              + id + ")};\n";
    }
    if (not kernel.uniforms.empty()) {
        text += "    program.uniforms = tc::Span<tc::UniformSlot const>{kUniforms" + id + ", std::size(kUniforms" + id
              + ")};\n";
    }
    text += "    program.pushConstantSize = " + std::to_string(kernel.pushConstantSize) + ";\n";
    text += "    return program;\n}()};\n\n";
    return text;
}

std::string escapeMakePath(std::string const &path) {
    auto escaped = std::string{};
    for (auto const c : path) {
        if (c == ' ' or c == '#') {
            escaped += '\\';
        } else if (c == '$') {
            escaped += '$';
        }
        escaped += c;
    }
    return escaped;
}

} // namespace

std::string writeRegistrationSource(std::vector<KernelArtifact> const &kernels, std::string const &sourceName) {
    auto text = "// Generated by tc-transpile from " + sourceName + ". Do not edit.\n";
    text += "#include <cstdint>\n#include <iterator>\n#include <string_view>\n\n#include <tc/program.hpp>\n\n";
    text += "namespace {\n\n";
    for (std::size_t i = 0; i < kernels.size(); ++i) {
        text += kernelBlock(kernels[i], i);
    }
    return text + "} // namespace\n";
}

std::string writeDepfile(std::string const &target, std::vector<std::string> const &dependencies) {
    auto text = escapeMakePath(target) + ":";
    for (auto const &dependency : dependencies) {
        text += " \\\n  " + escapeMakePath(dependency);
    }
    return text + "\n";
}

} // namespace tc::transpiler
