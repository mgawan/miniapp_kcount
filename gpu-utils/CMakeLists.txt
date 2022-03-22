cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gpu_utils LANGUAGES CXX)
set(CMAKE_CUDA_STANDARD 17)

set (MY_PLATFORM $ENV{HIP_PLATFORM})
message (STATUS "HIP_PLATFORM is set to: ${MY_PLATFORM}")

if(NOT MY_PLATFORM STREQUAL "amd")
    if(MY_PLATFORM STREQUAL "cuda")
        message(STATUS "Building for CUDA devices")
	project(gpu_utils LANGUAGES CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    endif()
else()
     message(STATUS "Building for AMD devices")
    find_package(hip)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB SOURCE_FILES *.cpp)
file(GLOB HEADER_FILES *.hpp)

if(NOT MY_PLATFORM STREQUAL "amd")
	if(MY_PLATFORM STREQUAL "cuda")
		enable_language(CUDA)
		set_source_files_properties(${SOURCE_FILES} PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE CUDA)
	endif()
endif()


add_library(utils_obj OBJECT ${SOURCE_FILES})

if(MY_PLATFORM STREQUAL "amd")
    target_compile_definitions(utils_obj PUBLIC PLATFORM_AMD=true)
endif()

add_library(utils_lib_shared SHARED $<TARGET_OBJECTS:utils_obj>)
if(NOT MY_PLATFORM STREQUAL "amd")
    if(MY_PLATFORM STREQUAL "cuda")
	set_property(TARGET utils_obj utils_lib_shared PROPERTY CUDA_ARCHITECTURES 50 60 70 80)
    endif()
endif()

target_link_libraries(utils_lib_shared hip::device)

