find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # Cross builds (Android) have no system GoogleTest; build it from source.
    include(FetchContent)
    FetchContent_Declare(googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
        URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(tc_core_tests core_tests.cpp cpu_executor_tests.cpp hazard_tests.cpp)
target_link_libraries(tc_core_tests PRIVATE tc tc_cpu GTest::gtest_main tc_warnings)
add_test(NAME tc_core_tests COMMAND tc_core_tests)

add_executable(tc_equivalence_tests
    equivalence.hpp
    equivalence_tests.cpp
    dispatch_tests.cpp
    texture_tests.cpp
    submission_tests.cpp
    interop_tests.cpp
    subgroup_tests.cpp
    reload_tests.cpp
    compressed_tests.cpp
    memory_tests.cpp
    transfer_tests.cpp
    kernels/feature_kernels.hpp
    kernels/dispatch_kernels.hpp
    kernels/texture_kernels.hpp
    kernels/subgroup_kernels.hpp
    kernels/memory_kernels.hpp)
target_include_directories(tc_equivalence_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/examples ${PROJECT_SOURCE_DIR}/examples/common/include)
# A plain Vulkan device exports the memory the import tests share.
if(TC_ENABLE_VULKAN)
    find_package(Vulkan REQUIRED)
    target_sources(tc_equivalence_tests PRIVATE external_memory_tests.cpp)
    target_link_libraries(tc_equivalence_tests PRIVATE Vulkan::Vulkan)
endif()
target_link_libraries(tc_equivalence_tests PRIVATE tc_devices tc_cpu GTest::gtest_main tc_warnings
    astcenc-none-static) # the compressor makes ASTC test data
tc_add_kernels(tc_equivalence_tests HEADERS
    kernels/feature_kernels.hpp
    kernels/dispatch_kernels.hpp
    kernels/texture_kernels.hpp
    kernels/subgroup_kernels.hpp
    kernels/memory_kernels.hpp
    ${PROJECT_SOURCE_DIR}/examples/game_of_life/game_of_life.hpp
    ${PROJECT_SOURCE_DIR}/examples/raytracer/raytracer.hpp
    ${PROJECT_SOURCE_DIR}/examples/nv12_to_rgba/nv12_to_rgba.hpp
    ${PROJECT_SOURCE_DIR}/examples/external_frame/external_frame.hpp)
find_program(TC_GLSLANG_VALIDATOR glslangValidator)
target_compile_definitions(tc_equivalence_tests PRIVATE
    TC_TEST_GLSL_DIR="${CMAKE_CURRENT_BINARY_DIR}/tc_generated/tc_equivalence_tests/glsl"
    TC_TEST_GLSLANG="${TC_GLSLANG_VALIDATOR}")
add_test(NAME tc_equivalence_tests COMMAND tc_equivalence_tests)

# Each transpiler/rejects_*.hpp starts with `// expect: <regex>`, the error
# tc-transpile must report for it.
if(TARGET tc_transpile)
    file(GLOB rejection_cases CONFIGURE_DEPENDS transpiler/rejects_*.hpp)
    foreach(case IN LISTS rejection_cases)
        get_filename_component(name ${case} NAME_WE)
        file(STRINGS ${case} expected REGEX "^// expect: " LIMIT_COUNT 1)
        string(REPLACE "// expect: " "" expected "${expected}")
        add_test(NAME transpiler_${name}
                 COMMAND tc_transpile ${case}
                         --output ${CMAKE_CURRENT_BINARY_DIR}/${name}.tc.cpp
                         --glsl-dir ${CMAKE_CURRENT_BINARY_DIR}/${name}_glsl
                         -- -std=c++17 -I${PROJECT_SOURCE_DIR}/tc/include)
        set_tests_properties(transpiler_${name} PROPERTIES PASS_REGULAR_EXPRESSION "${expected}")
    endforeach()
endif()
