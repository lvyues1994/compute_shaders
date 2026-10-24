add_subdirectory(cpu)

if(TC_ENABLE_OPENGL)
    add_subdirectory(opengl)
endif()

if(TC_ENABLE_VULKAN)
    add_subdirectory(vulkan)
endif()

add_subdirectory(devices)
