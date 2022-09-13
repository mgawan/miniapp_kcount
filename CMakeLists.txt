cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(app_gpu LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

# Set a default build type if none was specified
set(default_build_type "Release")
 
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

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
