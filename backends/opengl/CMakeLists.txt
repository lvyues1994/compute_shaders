find_path(TC_GLES3_INCLUDE_DIR GLES3/gl32.h REQUIRED)
find_library(TC_EGL_LIBRARY EGL REQUIRED)

file(GLOB_RECURSE tc_opengl_sources CONFIGURE_DEPENDS src/*.cpp src/*.hpp include/*.hpp)

add_library(tc_opengl STATIC ${tc_opengl_sources})
target_include_directories(tc_opengl PUBLIC include PRIVATE ${TC_GLES3_INCLUDE_DIR})
target_link_libraries(tc_opengl PUBLIC tc PRIVATE ${TC_EGL_LIBRARY} ${CMAKE_DL_LIBS} tc_warnings)

if(ANDROID)
    find_library(TC_GLESV3_LIBRARY GLESv3 REQUIRED)
    target_link_libraries(tc_opengl PRIVATE ${TC_GLESV3_LIBRARY})
endif()
