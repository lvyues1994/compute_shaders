find_package(GTest REQUIRED)

add_executable(tc_core_tests core_tests.cpp)
target_link_libraries(tc_core_tests PRIVATE tc tc_cpu GTest::gtest_main tc_warnings)
add_test(NAME tc_core_tests COMMAND tc_core_tests)

add_executable(tc_equivalence_tests equivalence_tests.cpp kernels/feature_kernels.hpp)
target_include_directories(tc_equivalence_tests PRIVATE ${PROJECT_SOURCE_DIR}/examples)
target_link_libraries(tc_equivalence_tests PRIVATE tc_devices tc_cpu GTest::gtest_main tc_warnings)
tc_add_kernels(tc_equivalence_tests HEADERS
    kernels/feature_kernels.hpp
    ${PROJECT_SOURCE_DIR}/examples/game_of_life/game_of_life.hpp
    ${PROJECT_SOURCE_DIR}/examples/raytracer/raytracer.hpp)
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
