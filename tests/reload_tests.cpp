// Kernels edited on disk while the program runs (ProgramRegistry::reloadFrom).

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#include <unistd.h>

#include <tc/program.hpp>

#include "equivalence.hpp"
#include "kernels/dispatch_kernels.hpp"

namespace {

using namespace tc;
using namespace tc::testing_support;
namespace fs = std::filesystem;

std::string read(fs::path const &path) {
    auto file = std::ifstream{path, std::ios::binary};
    return std::string{std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

void write(fs::path const &path, std::string const &text) {
    auto file = std::ofstream{path, std::ios::binary | std::ios::trunc};
    file << text;
}

void replaceAll(std::string &text, std::string const &from, std::string const &to) {
    for (auto at = text.find(from); at != std::string::npos; at = text.find(from, at + to.size())) {
        text.replace(at, from.size(), to);
    }
}

float runReloadMe(Device &device) {
    auto out = BufferResource<float>{64};
    auto kernel = tests::ReloadMe{};
    kernel.out.attach(out);
    device.uploadBuffer(out);
    device.useKernel(kernel);
    device.bindBuffer(kernel.out);
    device.execute(kernel, uvec3{1u, 1u, 1u});
    device.downloadBuffer(out);
    return out[10];
}

TEST_P(Equivalence, ReloadEditedKernels) {
    auto const generated = fs::path{TC_TEST_GLSL_DIR};
    if (not fs::exists(generated / "reload_me.interface")) {
        GTEST_SKIP() << "the generated GLSL is not on this machine";
    }
    auto const edited = fs::temp_directory_path() / ("tc_reload_" + std::to_string(getpid()));
    fs::create_directories(edited);
    for (auto const *suffix : {".gl.comp", ".es.comp", ".vk.comp", ".interface"}) {
        auto text = read(generated / (std::string{"reload_me"} + suffix));
        replaceAll(text, "= 1.0;", "= 2.0;");
        write(edited / (std::string{"reload_me"} + suffix), text);
    }
    auto const compile = std::string{TC_TEST_GLSLANG} + " -V --target-env vulkan1.0 -o " +
                         (edited / "reload_me.vk.comp.spv").string() + " " + (edited / "reload_me.vk.comp").string() +
                         " > /dev/null";
    ASSERT_EQ(std::system(compile.c_str()), 0);

    EXPECT_EQ(runReloadMe(*gpu), 1.0f);
    auto const report = ProgramRegistry::instance().reloadFrom(edited.string());
    ASSERT_EQ(report.reloaded, std::vector<std::string>{"reload_me"});
    EXPECT_TRUE(report.rejected.empty());
    EXPECT_EQ(ProgramRegistry::instance().find("reload_me")->revision % 2, 1u);
    EXPECT_EQ(runReloadMe(*gpu), 2.0f) << "the device picked up the edited kernel";
    EXPECT_EQ(runReloadMe(*cpu), 1.0f) << "the CPU runs the compiled C++";
    EXPECT_TRUE(ProgramRegistry::instance().reloadFrom(edited.string()).reloaded.empty()) << "nothing new";

    // A new binding would not match the host code: the edit is refused.
    write(edited / "reload_me.interface", read(edited / "reload_me.interface") + "resource 1 kind 0 access 0 extra\n");
    auto const refused = ProgramRegistry::instance().reloadFrom(edited.string());
    ASSERT_EQ(refused.rejected.size(), 1u);
    EXPECT_EQ(refused.rejected[0].program, "reload_me");
    EXPECT_EQ(runReloadMe(*gpu), 2.0f);

    // Back to what the build produced.
    EXPECT_EQ(ProgramRegistry::instance().reloadFrom(generated.string()).reloaded, std::vector<std::string>{"reload_me"});
    EXPECT_EQ(runReloadMe(*gpu), 1.0f);
    fs::remove_all(edited);
}

} // namespace
