add_library(example_cli INTERFACE)
target_include_directories(example_cli INTERFACE common/include)
target_link_libraries(example_cli INTERFACE tc_devices)

function(tc_add_example name)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "KERNELS")
    add_executable(${name} ${name}/main.cpp ${arg_KERNELS})
    target_link_libraries(${name} PRIVATE example_cli tc_warnings)
    tc_add_kernels(${name} HEADERS ${arg_KERNELS})
endfunction()

tc_add_example(float_adder KERNELS float_adder/float_adder.hpp)
tc_add_example(game_of_life KERNELS game_of_life/game_of_life.hpp)
tc_add_example(raytracer KERNELS raytracer/raytracer.hpp)
