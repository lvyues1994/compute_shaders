file(GLOB_RECURSE tc_devices_sources CONFIGURE_DEPENDS src/*.cpp include/*.hpp)

add_library(tc_devices STATIC ${tc_devices_sources})
target_include_directories(tc_devices PUBLIC include)
target_link_libraries(tc_devices PUBLIC tc PRIVATE tc_cpu tc_warnings)

if(TARGET tc_opengl)
    target_link_libraries(tc_devices PRIVATE tc_opengl)
    target_compile_definitions(tc_devices PRIVATE TC_HAS_OPENGL=1)
endif()

if(TARGET tc_vulkan)
    target_link_libraries(tc_devices PRIVATE tc_vulkan)
    target_compile_definitions(tc_devices PRIVATE TC_HAS_VULKAN=1)
endif()
