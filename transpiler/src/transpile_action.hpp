#pragma once

#include <memory>
#include <string>
#include <vector>

#include <clang/Tooling/Tooling.h>

#include "kernel_model.hpp"

namespace tc::transpiler {

// What the Clang pass produced for one input header.
struct TranspileResult {
    std::vector<KernelArtifact> kernels;
    std::vector<std::string> dependencies;
    bool ok = false;
};

// Front-end action factory: parses the header, finds and validates kernels,
// and fills `result` with their GLSL and reflection data.
std::unique_ptr<clang::tooling::FrontendActionFactory> makeTranspileActionFactory(TranspileResult &result,
                                                                                  std::string sourceName);

} // namespace tc::transpiler
