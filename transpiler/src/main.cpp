// tc-transpile: turns the kernel structs of one C++ header into GLSL for
// OpenGL, OpenGL ES and Vulkan (SPIR-V), plus a C++ source registering them.
//
//   tc-transpile kernels.hpp --output kernels.tc.cpp --glsl-dir glsl/
//                [--depfile kernels.tc.cpp.d] [--glslang glslangValidator]
//                -- -std=c++17 -I<include dirs>

#include <cstdlib>
#include <string>

#include <clang/Tooling/ArgumentsAdjusters.h>
#include <clang/Tooling/CommonOptionsParser.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>

#include "glslang_runner.hpp"
#include "registration_writer.hpp"
#include "transpile_action.hpp"

namespace {

using namespace tc::transpiler;

llvm::cl::OptionCategory category{"tc-transpile options"};
llvm::cl::opt<std::string> outputPath{"output", llvm::cl::desc("Generated registration C++ source"),
                                      llvm::cl::value_desc("file"), llvm::cl::Required, llvm::cl::cat(category)};
llvm::cl::opt<std::string> glslDirectory{"glsl-dir", llvm::cl::desc("Directory for the generated .comp files"),
                                         llvm::cl::value_desc("dir"), llvm::cl::Required, llvm::cl::cat(category)};
llvm::cl::opt<std::string> depfilePath{"depfile", llvm::cl::desc("Write a Make-style dependency file"),
                                       llvm::cl::value_desc("file"), llvm::cl::cat(category)};
llvm::cl::opt<std::string> glslangPath{"glslang", llvm::cl::desc("glslangValidator executable"),
                                       llvm::cl::value_desc("path"), llvm::cl::init("glslangValidator"),
                                       llvm::cl::cat(category)};

bool writeFile(std::string const &path, std::string const &contents) {
    auto error = std::error_code{};
    auto stream = llvm::raw_fd_ostream{path, error, llvm::sys::fs::OF_Text};
    if (error) {
        llvm::errs() << "error: cannot write " << path << ": " << error.message() << "\n";
        return false;
    }
    stream << contents;
    return true;
}

std::string glslPath(KernelArtifact const &kernel, char const *dialect) {
    auto path = llvm::SmallString<256>{glslDirectory.getValue()};
    llvm::sys::path::append(path, kernel.name + "." + dialect + ".comp");
    return std::string{path};
}

// Writes the three GLSL variants, validates them with glslangValidator and
// compiles the Vulkan one to SPIR-V. ES failures only disable ES support.
bool compileShaders(KernelArtifact &kernel, GlslangRunner const &glslang) {
    auto const desktop = glslPath(kernel, "gl");
    auto const es = glslPath(kernel, "es");
    auto const vulkan = glslPath(kernel, "vk");
    if (not writeFile(desktop, kernel.glslDesktop) or not writeFile(vulkan, kernel.glslVulkan)) {
        return false;
    }
    auto const desktopResult = glslang.validate(desktop);
    if (not desktopResult.ok) {
        llvm::errs() << desktop << ": error: generated OpenGL GLSL does not compile:\n" << desktopResult.log << "\n";
        return false;
    }
    auto spirv = glslang.compileSpirv(vulkan, vulkan + ".spv");
    if (not spirv.ok) {
        llvm::errs() << vulkan << ": error: generated Vulkan GLSL does not compile:\n" << spirv.log << "\n";
        return false;
    }
    kernel.spirv = std::move(spirv.spirv);
    if (kernel.glslEs.empty()) {
        llvm::errs() << "note: kernel '" << kernel.name << "' has no OpenGL ES variant: " << kernel.esUnavailableReason
                     << "\n";
        return true;
    }
    if (not writeFile(es, kernel.glslEs)) {
        return false;
    }
    auto const esResult = glslang.validate(es);
    if (not esResult.ok) {
        llvm::errs() << es << ": warning: OpenGL ES variant rejected, ES support disabled:\n" << esResult.log << "\n";
        kernel.esUnavailableReason = "glslangValidator rejected the OpenGL ES variant";
        kernel.glslEs.clear();
    }
    return true;
}

std::string resolveGlslang() {
    if (llvm::sys::fs::can_execute(glslangPath.getValue())) {
        return glslangPath.getValue();
    }
    if (auto const found = llvm::sys::findProgramByName(glslangPath.getValue())) {
        return *found;
    }
    return {};
}

} // namespace

int main(int argc, char const **argv) {
    auto parser = clang::tooling::CommonOptionsParser::create(argc, argv, category, llvm::cl::Required);
    if (not parser) {
        llvm::errs() << llvm::toString(parser.takeError()) << "\n";
        return EXIT_FAILURE;
    }
    auto const &sources = parser->getSourcePathList();
    if (sources.size() != 1) {
        llvm::errs() << "error: tc-transpile takes exactly one kernel header\n";
        return EXIT_FAILURE;
    }
    llvm::sys::fs::remove(outputPath.getValue());

    auto tool = clang::tooling::ClangTool{parser->getCompilations(), sources};
    tool.clearArgumentsAdjusters();
    tool.appendArgumentsAdjuster(clang::tooling::getClangSyntaxOnlyAdjuster());
    tool.appendArgumentsAdjuster(clang::tooling::getInsertArgumentAdjuster(
        {"-xc++", "-Wno-pragma-once-outside-header", "-resource-dir", TC_CLANG_RESOURCE_DIR},
        clang::tooling::ArgumentInsertPosition::BEGIN));

    auto const sourceName = llvm::sys::path::filename(sources.front()).str();
    auto result = TranspileResult{};
    auto const factory = makeTranspileActionFactory(result, sourceName);
    if (tool.run(factory.get()) != 0 or not result.ok) {
        return EXIT_FAILURE;
    }

    auto const glslang = GlslangRunner{resolveGlslang()};
    if (glslang.executable.empty()) {
        llvm::errs() << "error: glslangValidator not found (pass --glslang <path>)\n";
        return EXIT_FAILURE;
    }
    if (auto const error = llvm::sys::fs::create_directories(glslDirectory.getValue())) {
        llvm::errs() << "error: cannot create " << glslDirectory << ": " << error.message() << "\n";
        return EXIT_FAILURE;
    }
    for (auto &kernel : result.kernels) {
        if (not compileShaders(kernel, glslang)) {
            return EXIT_FAILURE;
        }
    }
    if (not depfilePath.empty() and not writeFile(depfilePath, writeDepfile(outputPath, result.dependencies))) {
        return EXIT_FAILURE;
    }
    auto headerPath = llvm::SmallString<256>{sources.front()};
    llvm::sys::fs::make_absolute(headerPath);
    if (not writeFile(outputPath, writeRegistrationSource(result.kernels, sourceName, headerPath.str().str()))) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
