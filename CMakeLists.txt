cmake_minimum_required(VERSION 3.15)
project(VectorHub)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
include(CheckCXXCompilerFlag)

# xsimd: header-only SIMD wrapper used to vectorize distance computations.
# Fetched by URL (not GIT_REPOSITORY) since this environment has no git.
set(XSIMD_SKIP_INSTALL ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    xsimd
    URL https://github.com/xtensor-stack/xsimd/archive/refs/tags/13.2.0.tar.gz
    URL_HASH SHA256=edd8cd3d548c185adc70321c53c36df41abe64c1fe2c67bc6d93c3ecda82447a
)
FetchContent_MakeAvailable(xsimd)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# Only enable AVX2+FMA if the compiler actually supports it; falls back to
# whatever baseline xsimd picks (e.g. SSE2) otherwise.
check_cxx_compiler_flag("-mavx2 -mfma" HNSW_HAS_AVX2_FMA)

# HNSW C++ library (object library to be linked with extension)
add_library(hnsw_lib OBJECT
    cpp/src/hnsw/hnsw.cpp
)
target_include_directories(hnsw_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src)
target_link_libraries(hnsw_lib PUBLIC xsimd)
if(HNSW_HAS_AVX2_FMA)
    target_compile_options(hnsw_lib PRIVATE -mavx2 -mfma)
endif()

# Python extension module
pybind11_add_module(_vectorhub
    cpp/pybind11_binding/binding.cpp
    $<TARGET_OBJECTS:hnsw_lib>
)

# Include directories
target_include_directories(_vectorhub PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src)

# Install the extension into the vectorhub package
install(TARGETS _vectorhub DESTINATION vectorhub)
