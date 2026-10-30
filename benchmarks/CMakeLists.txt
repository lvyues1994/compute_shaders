add_executable(tc_bench main.cpp harness.hpp kernels.hpp)
target_include_directories(tc_bench PRIVATE
    ${PROJECT_SOURCE_DIR}/examples ${PROJECT_SOURCE_DIR}/examples/common/include)
target_link_libraries(tc_bench PRIVATE tc_devices tc_warnings)
tc_add_kernels(tc_bench HEADERS
    kernels.hpp
    ${PROJECT_SOURCE_DIR}/examples/nv12_to_rgba/nv12_to_rgba.hpp
    ${PROJECT_SOURCE_DIR}/examples/game_of_life/game_of_life.hpp)
