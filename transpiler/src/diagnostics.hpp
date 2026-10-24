#pragma once

#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/SourceLocation.h>
#include <llvm/ADT/StringRef.h>

namespace tc::transpiler {

// Reports through Clang's diagnostics engine, so messages carry
// `file:line:col: error:` prefixes that IDEs and build logs understand.
struct Diagnostics {
    explicit Diagnostics(clang::DiagnosticsEngine &engine_)
        : engine(engine_),
          errorId(engine_.getCustomDiagID(clang::DiagnosticsEngine::Error, "%0")),
          warningId(engine_.getCustomDiagID(clang::DiagnosticsEngine::Warning, "%0")),
          noteId(engine_.getCustomDiagID(clang::DiagnosticsEngine::Note, "%0")) {}

    void error(clang::SourceLocation const loc, llvm::StringRef const message) {
        ++errorCount;
        engine.Report(loc, errorId) << message;
    }

    void warning(clang::SourceLocation const loc, llvm::StringRef const message) {
        engine.Report(loc, warningId) << message;
    }

    void note(clang::SourceLocation const loc, llvm::StringRef const message) {
        engine.Report(loc, noteId) << message;
    }

    bool hasErrors() const noexcept { return errorCount != 0; }
    unsigned errors() const noexcept { return errorCount; }

private:
    clang::DiagnosticsEngine &engine;
    unsigned errorId;
    unsigned warningId;
    unsigned noteId;
    unsigned errorCount = 0;
};

} // namespace tc::transpiler
