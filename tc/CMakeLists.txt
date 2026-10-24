file(GLOB_RECURSE tc_sources CONFIGURE_DEPENDS src/*.cpp include/*.hpp)

add_library(tc STATIC ${tc_sources})
target_include_directories(tc PUBLIC include)
target_link_libraries(tc PRIVATE tc_warnings)
if(ANDROID)
    target_link_libraries(tc PUBLIC nativewindow)
endif()
