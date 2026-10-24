find_package(Vulkan REQUIRED)

file(GLOB_RECURSE tc_vulkan_sources CONFIGURE_DEPENDS src/*.cpp src/*.hpp include/*.hpp)

add_library(tc_vulkan STATIC ${tc_vulkan_sources})
target_include_directories(tc_vulkan PUBLIC include)
target_link_libraries(tc_vulkan PUBLIC tc PRIVATE Vulkan::Vulkan tc_warnings)
if(ANDROID)
    target_compile_definitions(tc_vulkan PRIVATE VK_USE_PLATFORM_ANDROID_KHR)
endif()
