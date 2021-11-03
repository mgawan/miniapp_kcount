cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(app_gpu LANGUAGES CXX)
set(CMAKE_CUDA_STANDARD 11)


include_directories("parse_and_pack" "./parse_and_pack/gpu-utils")

# Add adept library code
add_subdirectory(parse_and_pack)

add_executable(main_app main.cpp)
target_link_libraries(main_app pnp_lib_shared)
