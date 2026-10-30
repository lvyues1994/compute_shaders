file(GLOB_RECURSE tc_sources CONFIGURE_DEPENDS src/*.cpp include/*.hpp)

add_library(tc STATIC ${tc_sources})
target_include_directories(tc PUBLIC include)
target_link_libraries(tc PRIVATE tc_warnings)
if(TC_PROGRAM_RELOAD)
    target_compile_definitions(tc PRIVATE TC_PROGRAM_RELOAD=1)
endif()
if(ANDROID)
    target_link_libraries(tc PUBLIC nativewindow)
endif()

# Decoders of compressed texture formats, for the CPU backend: bcdec (BC1-7)
# and etcdec (ETC2, EAC), single headers under MIT / Unlicense, and Arm's
# astc-encoder (Apache-2.0, ASTC), whose compressor the tests also use.
include(FetchContent)
FetchContent_Declare(tc_bcdec
    URL https://raw.githubusercontent.com/iOrange/bcdec/80859ed3b7afb1c527a2a99d70c61457bea72d0c/bcdec.h
    URL_HASH SHA256=134520764d96f70a27db814173616c89ad85db95fbd3328417c7a8c77e7ca189
    DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_Declare(tc_etcdec
    URL https://raw.githubusercontent.com/iOrange/etcdec/972875d403ed8ac27e0f35c2f29d819e710a688a/etcdec.h
    URL_HASH SHA256=57317112d44a1fc89d153fa4141587963a2ceeb22c597f23c594b301e1de6838
    DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_Declare(tc_astcenc
    URL https://github.com/ARM-software/astc-encoder/archive/refs/tags/5.7.0.tar.gz
    URL_HASH SHA256=7c1b28ece59c9c2737e297123a9910c1c548676c7aed87a09d5734e2fa7bdaf0
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(ASTCENC_ISA_NONE ON CACHE BOOL "" FORCE)
set(ASTCENC_CLI OFF CACHE BOOL "" FORCE)
set(ASTCENC_UNITTEST OFF CACHE BOOL "" FORCE)
set(ASTCENC_SHAREDLIB OFF CACHE BOOL "" FORCE)
set(ASTCENC_WERROR OFF CACHE BOOL "" FORCE)
set(ASTCENC_UNIVERSAL_BUILD OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(tc_bcdec tc_etcdec tc_astcenc)
set_target_properties(astcenc-none-static PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(tc PRIVATE ${tc_bcdec_SOURCE_DIR} ${tc_etcdec_SOURCE_DIR})
target_link_libraries(tc PRIVATE astcenc-none-static)
