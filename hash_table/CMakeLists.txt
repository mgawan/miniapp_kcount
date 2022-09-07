cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(gpu_ht LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

set (MY_PLATFORM $ENV{HIP_PLATFORM})
message (STATUS "HIP_PLATFORM is set to: ${MY_PLATFORM}")


if(MY_PLATFORM STREQUAL "cuda")
    message(STATUS "Building for CUDA devices")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
elseif(MY_PLATFORM STREQUAL "amd")
     message(STATUS "Building for AMD devices")
     find_package(hip)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(../CMakeDefinitions.txt)

file(GLOB SOURCE_FILES *.cpp)
file(GLOB HEADER_FILES *.hpp)


if(MY_PLATFORM STREQUAL "cuda")
	enable_language(CUDA)
	set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE CUDA)
elseif(MY_PLATFORM STREQUAL "amd")
    enable_language(HIP)
    set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE HIP LINKER_LANGUAGE HIP)
endif()

add_library(gpu_ht_obj OBJECT ${SOURCE_FILES})

if(MY_PLATFORM STREQUAL "amd")
    target_compile_definitions(gpu_ht_obj PUBLIC PLATFORM_AMD=true)
endif()

add_library(gpu_ht_lib_shared SHARED $<TARGET_OBJECTS:gpu_ht_obj>)
if(NOT MY_PLATFORM STREQUAL "amd")
    if(MY_PLATFORM STREQUAL "cuda")
	set_property(TARGET gpu_ht_obj gpu_ht_lib_shared PROPERTY CUDA_ARCHITECTURES 50 60 70 80)
    endif()
endif()

target_link_libraries(gpu_ht_lib_shared utils_lib_shared)

