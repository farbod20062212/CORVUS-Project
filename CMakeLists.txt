cmake_minimum_required(VERSION 3.10...3.31)
project(minnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)      # lets IDEs / clangd find include paths (compile_commands.json)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MINNN_NATIVE    "Optimise for this CPU (-march=native). Turn OFF for binaries you copy to other machines." ON)
option(MINNN_FETCH_STB "Download include/third_party/stb_image.h automatically if it is missing." ON)

# ---------------------------------------------------------------------------
# Pre-flight: fail early with a readable message instead of a cryptic compiler error.
# ---------------------------------------------------------------------------
set(MINNN_FILES
  src/main.cpp src/stb_impl.cpp
  include/tensor.hpp include/layers.hpp include/loss.hpp include/optimizer.hpp
  include/dataset.hpp include/text.hpp include/image.hpp include/imageproc.hpp
  include/model.hpp include/trainer.hpp include/bpe.hpp include/transformer.hpp
  include/lm.hpp include/command.hpp include/selftest.hpp)
set(MINNN_MISSING "")
foreach(f ${MINNN_FILES})
  if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${f}")
    list(APPEND MINNN_MISSING ${f})
  endif()
endforeach()
if(MINNN_MISSING)
  string(REPLACE ";" "\n  " MINNN_MISSING_TXT "${MINNN_MISSING}")
  message(FATAL_ERROR
    "minnn: these project files are missing (relative to ${CMAKE_CURRENT_SOURCE_DIR}):\n  ${MINNN_MISSING_TXT}\n"
    "Copy the whole project folder (keep the src/ and include/ sub-folders), not single files.")
endif()

set(MINNN_STB "${CMAKE_CURRENT_SOURCE_DIR}/include/third_party/stb_image.h")
if(NOT EXISTS "${MINNN_STB}")
  if(MINNN_FETCH_STB)
    message(STATUS "stb_image.h not found - downloading it (public domain, one file)")
    file(DOWNLOAD "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h" "${MINNN_STB}"
         STATUS MINNN_DL_STATUS TLS_VERIFY ON)
    list(GET MINNN_DL_STATUS 0 MINNN_DL_CODE)
    if(NOT MINNN_DL_CODE EQUAL 0)
      file(REMOVE "${MINNN_STB}")
      message(FATAL_ERROR "minnn: could not download stb_image.h (${MINNN_DL_STATUS}).\n"
        "Download https://raw.githubusercontent.com/nothings/stb/master/stb_image.h by hand into\n"
        "  ${CMAKE_CURRENT_SOURCE_DIR}/include/third_party/")
    endif()
  else()
    message(FATAL_ERROR "minnn: include/third_party/stb_image.h is missing (set -DMINNN_FETCH_STB=ON to download it).")
  endif()
endif()

# ---------------------------------------------------------------------------
add_executable(minnn src/main.cpp src/stb_impl.cpp)
target_include_directories(minnn PRIVATE include)

# Compiler-specific flags, only when the compiler understands them (GCC / Clang / MSVC).
include(CheckCXXCompilerFlag)
if(MSVC)
  target_compile_options(minnn PRIVATE $<$<CONFIG:Release>:/O2>)
else()
  target_compile_options(minnn PRIVATE $<$<CONFIG:Release>:-O3>)
  if(MINNN_NATIVE)
    check_cxx_compiler_flag("-march=native" MINNN_HAS_MARCH_NATIVE)
    if(MINNN_HAS_MARCH_NATIVE)
      target_compile_options(minnn PRIVATE $<$<CONFIG:Release>:-march=native>)
    endif()
  endif()
endif()

# OpenMP is optional: without it everything still builds and runs (single-threaded).
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(minnn PRIVATE OpenMP::OpenMP_CXX)
else()
  message(STATUS "OpenMP not found - building single-threaded (install libomp / a compiler with -fopenmp for speed)")
endif()

include(CTest)
if(BUILD_TESTING)
  add_test(NAME selftest COMMAND minnn selftest)
endif()
