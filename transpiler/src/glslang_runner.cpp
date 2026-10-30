#include "glslang_runner.hpp"

#include <cstring>
#include <optional>

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Program.h>

namespace tc::transpiler {

GlslangResult GlslangRunner::run(std::vector<std::string> const &args, std::string const &logPath) const {
    auto argv = llvm::SmallVector<llvm::StringRef, 8>{executable};
    for (auto const &arg : args) {
        argv.push_back(arg);
    }
    std::optional<llvm::StringRef> const redirects[] = {std::nullopt, llvm::StringRef{logPath},
                                                         llvm::StringRef{logPath}};
    auto error = std::string{};
    auto const status = llvm::sys::ExecuteAndWait(executable, argv, std::nullopt, redirects, 0, 0, &error);
    auto result = GlslangResult{};
    if (auto log = llvm::MemoryBuffer::getFile(logPath)) {
        result.log = (*log)->getBuffer().str();
    }
    llvm::sys::fs::remove(logPath);
    if (status < 0) {
        result.log = "cannot run " + executable + ": " + error;
        return result;
    }
    result.ok = status == 0;
    return result;
}

GlslangResult GlslangRunner::validate(std::string const &shaderPath) const {
    return run({shaderPath}, shaderPath + ".log");
}

GlslangResult GlslangRunner::compileSpirv(std::string const &shaderPath, std::string const &spirvPath,
                                         std::string const &targetEnvironment) const {
    auto result = run({"-V", "--target-env", targetEnvironment, "-o", spirvPath, shaderPath}, shaderPath + ".log");
    if (not result.ok) {
        return result;
    }
    auto binary = llvm::MemoryBuffer::getFile(spirvPath, false, false);
    if (not binary or (*binary)->getBufferSize() % 4 != 0) {
        result.ok = false;
        result.log += "\ncannot read SPIR-V output " + spirvPath;
        return result;
    }
    auto const bytes = (*binary)->getBuffer();
    result.spirv.resize(bytes.size() / 4);
    std::memcpy(result.spirv.data(), bytes.data(), bytes.size());
    return result;
}

} // namespace tc::transpiler
