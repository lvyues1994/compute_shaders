#pragma once

#include <string>
#include <vector>

#include "kernel_model.hpp"

namespace tc::transpiler {

// C++ source that embeds the kernels' GLSL, SPIR-V and reflection data and
// registers them with tc::ProgramRegistry at static-initialisation time.
std::string writeRegistrationSource(std::vector<KernelArtifact> const &kernels, std::string const &sourceName);

// Make-style dependency file: `target: dep1 dep2 ...`.
std::string writeDepfile(std::string const &target, std::vector<std::string> const &dependencies);

} // namespace tc::transpiler
