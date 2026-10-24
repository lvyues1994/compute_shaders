find_package(Threads REQUIRED)

file(GLOB_RECURSE tc_cpu_sources CONFIGURE_DEPENDS src/*.cpp include/*.hpp)

add_library(tc_cpu STATIC ${tc_cpu_sources})
target_include_directories(tc_cpu PUBLIC include)
target_link_libraries(tc_cpu PUBLIC tc PRIVATE Threads::Threads tc_warnings)
