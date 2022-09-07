cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(app_gpu LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(hip REQUIRED)
enable_language(HIP)

include(CMakeDefinitions.txt)
include_directories("parse_and_pack" "gpu-utils" "hash_table")

# Add adept library code
add_subdirectory(gpu-utils)
add_subdirectory(hash_table)
add_subdirectory(parse_and_pack)

add_executable(main_app main.cpp)

target_link_libraries(main_app pnp_lib_shared gpu_ht_lib_shared utils_lib_shared)
