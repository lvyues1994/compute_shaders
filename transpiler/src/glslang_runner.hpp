#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tc::transpiler {

struct GlslangResult {
    bool ok = false;
    std::string log;
    std::vector<std::uint32_t> spirv; // only filled by compileSpirv
};

// Runs the glslangValidator executable, as the talk does, to validate the
// generated GLSL and to compile the Vulkan variant to SPIR-V.
struct GlslangRunner {
    std::string executable;

    GlslangResult validate(std::string const &shaderPath) const;
    GlslangResult compileSpirv(std::string const &shaderPath, std::string const &spirvPath) const;

private:
    GlslangResult run(std::vector<std::string> const &args, std::string const &logPath) const;
};

} // namespace tc::transpiler
