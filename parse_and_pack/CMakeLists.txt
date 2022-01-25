cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pnp_gpu LANGUAGES CUDA)
set(CMAKE_CUDA_STANDARD 11)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
include(CMakeDefinitions.txt)

include_directories("gpu-utils")


add_subdirectory(gpu-utils)

file(GLOB SOURCE_FILES *.cpp)
file(GLOB HEADER_FILES *.hpp)

enable_language(CUDA)
set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE CUDA)

add_library(pnp_obj OBJECT ${SOURCE_FILES})
add_library(pnp_lib_shared SHARED $<TARGET_OBJECTS:pnp_obj>)
set_property(TARGET pnp_obj pnp_lib_shared PROPERTY CUDA_ARCHITECTURES 50 60 70 80)
target_link_libraries(pnp_lib_shared utils_lib_shared)