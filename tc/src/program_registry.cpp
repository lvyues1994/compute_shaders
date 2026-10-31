#include <tc/program.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>

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

std::vector<KernelProgram> ProgramRegistry::all() const {
    auto const lock = std::lock_guard<std::mutex>{mutex};
    auto result = std::vector<KernelProgram>{};
    result.reserve(programs.size());
    for (auto const &[name, program] : programs) {
        result.push_back(program);
    }
    return result;
}

std::string describeInterface(KernelProgram const &program) {
    auto text = std::ostringstream{};
    auto const size = program.localSize;
    text << "tc-interface 1\n";
    text << "local_size " << size.x << " " << size.y << " " << size.z << "\n";
    text << "push_constants " << program.pushConstantSize << "\n";
    text << "shared " << program.sharedBytes << "\n";
    text << "features " << program.features << "\n";
    for (auto const &slot : program.resources) {
        text << "resource " << slot.binding << " kind " << static_cast<int>(slot.kind) << " access "
             << static_cast<int>(slot.access);
        if (slot.kind == ResourceKind::StorageImage or slot.kind == ResourceKind::SampledTexture
            or slot.kind == ResourceKind::ExternalImage) {
            text << " dim " << static_cast<int>(slot.imageDim) << " format " << glslFormatQualifier(slot.imageFormat)
                 << " sampler " << static_cast<int>(slot.filter) << " " << static_cast<int>(slot.wrap) << " "
                 << static_cast<int>(slot.mip);
        }
        text << " " << slot.name << "\n";
    }
    for (auto const &slot : program.uniforms) {
        text << "uniform " << slot.location << " scalar " << static_cast<int>(slot.scalar) << " shape "
             << slot.components << "x" << slot.columns << " offset " << slot.pushConstantOffset << " " << slot.name
             << "\n";
    }
    return text.str();
}

#if defined(TC_PROGRAM_RELOAD)

namespace {

std::optional<std::string> readFile(std::string const &path) {
    auto file = std::ifstream{path, std::ios::binary};
    if (not file) {
        return std::nullopt;
    }
    return std::string{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

std::optional<std::vector<std::uint32_t>> readSpirv(std::string const &path) {
    auto const bytes = readFile(path);
    if (not bytes or bytes->size() % 4 != 0 or bytes->empty()) {
        return std::nullopt;
    }
    auto words = std::vector<std::uint32_t>(bytes->size() / 4);
    std::memcpy(words.data(), bytes->data(), bytes->size());
    return words;
}

bool sameWords(Span<std::uint32_t const> const a, std::vector<std::uint32_t> const &b) {
    return a.size() == b.size() and std::equal(a.begin(), a.end(), b.begin());
}

} // namespace

ReloadReport ProgramRegistry::reloadFrom(std::string const &directory) {
    auto const lock = std::lock_guard<std::mutex>{mutex};
    auto report = ReloadReport{};
    for (auto &[name, program] : programs) {
        auto const base = directory + "/" + std::string{name};
        auto const interface = readFile(base + ".interface");
        if (not interface) {
            continue; // not a kernel of this directory
        }
        if (*interface != describeInterface(program)) {
            report.rejected.push_back({std::string{name}, "its interface (bindings, access, uniforms, local size or "
                                                          "features) changed; rebuild and restart"});
            continue;
        }
        auto desktop = program.glslDesktop.empty() ? std::optional<std::string>{std::string{}} : readFile(base + ".gl.comp");
        auto spirv = readSpirv(base + ".vk.comp.spv");
        if (not desktop or not spirv) {
            report.rejected.push_back({std::string{name}, "its GLSL or SPIR-V is missing"});
            continue;
        }
        auto es = program.glslEs.empty() ? std::optional<std::string>{std::string{}} : readFile(base + ".es.comp");
        if (not es) {
            report.rejected.push_back({std::string{name}, "its OpenGL ES variant is missing"});
            continue;
        }
        if (*desktop == program.glslDesktop and *es == program.glslEs and sameWords(program.spirv, *spirv)) {
            continue;
        }
        auto sources = std::make_unique<ReloadedSources>(ReloadedSources{std::move(*desktop), std::move(*es), std::move(*spirv)});
        program.glslDesktop = sources->desktop;
        program.glslEs = sources->es;
        program.spirv = Span<std::uint32_t const>{sources->spirv.data(), sources->spirv.size()};
        ++program.revision;
        reloaded.push_back(std::move(sources));
        report.reloaded.emplace_back(name);
    }
    return report;
}

#else

ReloadReport ProgramRegistry::reloadFrom(std::string const &) {
    auto report = ReloadReport{};
    report.rejected.push_back({"*", "tc was built without TC_PROGRAM_RELOAD"});
    return report;
}

#endif

} // namespace tc
