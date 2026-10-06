# Optional ahead-of-time build of the kernels:
#   cmake -S quantization/kernels -B build && cmake --build build
#   QA_LIB=$PWD/build/libqa_kernels.so python ...
# Without it the library is compiled on first import (see native.py).
cmake_minimum_required(VERSION 3.15)
project(qa_kernels CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

add_library(qa_kernels SHARED qa_kernels.cpp)
set_target_properties(qa_kernels PROPERTIES OUTPUT_NAME qa_kernels)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(qa_kernels PRIVATE OpenMP::OpenMP_CXX)
endif()

# Portable by default; use -DQA_NATIVE_ARCH=ON for a machine-specific build.
option(QA_NATIVE_ARCH "Compile with -march=native" OFF)
if(QA_NATIVE_ARCH AND NOT MSVC)
  target_compile_options(qa_kernels PRIVATE -march=native)
endif()
