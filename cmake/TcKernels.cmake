# tc_add_kernels(<target> HEADERS <kernel headers...>)
#
# Runs tc-transpile on each header at build time. The generated source embeds
# the GLSL / SPIR-V of every kernel struct in the header and registers it with
# tc::ProgramRegistry; it is compiled into <target>. The .comp files land in
# <binary dir>/tc_generated/<target>/glsl for inspection.
#
# The transpiler is the in-tree tc_transpile target, or TC_TRANSPILER_EXECUTABLE
# (a host build) when cross-compiling, e.g. for Android.
function(tc_add_kernels target)
    cmake_parse_arguments(PARSE_ARGV 1 arg "" "" "HEADERS")
    if(TARGET tc_transpile)
        set(transpiler $<TARGET_FILE:tc_transpile>)
        set(transpiler_dependency tc_transpile)
    elseif(TC_TRANSPILER_EXECUTABLE)
        set(transpiler ${TC_TRANSPILER_EXECUTABLE})
        set(transpiler_dependency ${TC_TRANSPILER_EXECUTABLE})
    else()
        message(WARNING "tc_add_kernels(${target}): no tc-transpile available; "
                        "only the CPU backend can run these kernels")
        return()
    endif()
    find_program(TC_GLSLANG_VALIDATOR glslangValidator REQUIRED)

    set(out_dir ${CMAKE_CURRENT_BINARY_DIR}/tc_generated/${target})
    set(includes "$<TARGET_PROPERTY:${target},INCLUDE_DIRECTORIES>")
    foreach(header IN LISTS arg_HEADERS)
        get_filename_component(header_path ${header} ABSOLUTE)
        get_filename_component(header_dir ${header_path} DIRECTORY)
        get_filename_component(stem ${header} NAME_WE)
        set(out_cpp ${out_dir}/${stem}.tc.cpp)
        add_custom_command(
            OUTPUT ${out_cpp}
            COMMAND ${CMAKE_COMMAND} -E make_directory ${out_dir}
            COMMAND ${transpiler} ${header_path}
                    --output ${out_cpp}
                    --glsl-dir ${out_dir}/glsl
                    --depfile ${out_cpp}.d
                    --glslang ${TC_GLSLANG_VALIDATOR}
                    -- -std=c++17 -I${header_dir} "$<$<BOOL:${includes}>:-I$<JOIN:${includes},;-I>>"
            DEPENDS ${header_path} ${transpiler_dependency}
            DEPFILE ${out_cpp}.d
            COMMENT "tc-transpile ${header}"
            COMMAND_EXPAND_LISTS
            VERBATIM)
        target_sources(${target} PRIVATE ${out_cpp})
    endforeach()
endfunction()
