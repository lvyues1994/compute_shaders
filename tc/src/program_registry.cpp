#include <tc/program.hpp>

namespace tc {

ProgramRegistry &ProgramRegistry::instance() {
    static auto registry = ProgramRegistry{};
    return registry;
}

void ProgramRegistry::add(KernelProgram const &program) {
    auto const lock = std::lock_guard<std::mutex>{mutex};
    programs.insert_or_assign(program.name, program);
}

std::optional<KernelProgram> ProgramRegistry::find(std::string_view const name) const {
    auto const lock = std::lock_guard<std::mutex>{mutex};
    auto const it = programs.find(name);
    if (it == programs.end()) {
        return std::nullopt;
    }
    return it->second;
}

} // namespace tc
