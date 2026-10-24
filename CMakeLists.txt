cmake_minimum_required(VERSION 3.22)

project(tc_compute VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_CROSSCOMPILING)
    set(tc_default_transpiler OFF)
else()
    set(tc_default_transpiler ON)
endif()

option(TC_BUILD_TRANSPILER "Build the tc-transpile host tool (needs Clang/LLVM development files)"
       ${tc_default_transpiler})
option(TC_ENABLE_OPENGL "Build the OpenGL / OpenGL ES backend (EGL)" ON)
option(TC_ENABLE_VULKAN "Build the Vulkan backend" ON)
option(TC_BUILD_EXAMPLES "Build the examples" ${PROJECT_IS_TOP_LEVEL})
option(TC_BUILD_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})
set(TC_TRANSPILER_EXECUTABLE "" CACHE FILEPATH
    "Prebuilt host tc-transpile; used when TC_BUILD_TRANSPILER is OFF (e.g. Android cross builds)")

add_library(tc_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tc_warnings INTERFACE
        -Wall -Wextra -Wpedantic -Wnon-virtual-dtor -Werror=return-type)
endif()

add_subdirectory(tc)
add_subdirectory(backends)
if(TC_BUILD_TRANSPILER)
    add_subdirectory(transpiler)
endif()

include(cmake/TcKernels.cmake)

if(TC_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(TC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
