find_package(Clang REQUIRED CONFIG
    HINTS /usr/lib/llvm-18/lib/cmake/clang /usr/lib/llvm-19/lib/cmake/clang /usr/lib/llvm-20/lib/cmake/clang)
message(STATUS "tc-transpile: Clang ${LLVM_PACKAGE_VERSION} from ${Clang_DIR}")

file(GLOB_RECURSE tc_transpile_sources CONFIGURE_DEPENDS src/*.cpp src/*.hpp)

add_executable(tc_transpile ${tc_transpile_sources})
set_target_properties(tc_transpile PROPERTIES OUTPUT_NAME tc-transpile)
target_include_directories(tc_transpile SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
separate_arguments(tc_llvm_definitions NATIVE_COMMAND "${LLVM_DEFINITIONS}")
target_compile_definitions(tc_transpile PRIVATE
    ${tc_llvm_definitions}
    TC_CLANG_RESOURCE_DIR="${LLVM_LIBRARY_DIR}/clang/${LLVM_VERSION_MAJOR}")
if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(tc_transpile PRIVATE -fno-rtti)
endif()
target_link_libraries(tc_transpile PRIVATE clang-cpp LLVM tc tc_warnings)
