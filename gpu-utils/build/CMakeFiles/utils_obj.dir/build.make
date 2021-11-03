# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /global/homes/m/mgawan/mhm2_kcount/kcount_kernel_driver/gpu-utils

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /global/homes/m/mgawan/mhm2_kcount/kcount_kernel_driver/gpu-utils/build

# Include any dependencies generated for this target.
include CMakeFiles/utils_obj.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/utils_obj.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/utils_obj.dir/flags.make

CMakeFiles/utils_obj.dir/gpu_common.cpp.o: CMakeFiles/utils_obj.dir/flags.make
CMakeFiles/utils_obj.dir/gpu_common.cpp.o: ../gpu_common.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/homes/m/mgawan/mhm2_kcount/kcount_kernel_driver/gpu-utils/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/utils_obj.dir/gpu_common.cpp.o"
	/usr/common/software/sles15_cgpu/cuda/11.1.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /global/homes/m/mgawan/mhm2_kcount/kcount_kernel_driver/gpu-utils/gpu_common.cpp -o CMakeFiles/utils_obj.dir/gpu_common.cpp.o

CMakeFiles/utils_obj.dir/gpu_common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/utils_obj.dir/gpu_common.cpp.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/utils_obj.dir/gpu_common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/utils_obj.dir/gpu_common.cpp.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/utils_obj.dir/gpu_common.cpp.o.requires:

.PHONY : CMakeFiles/utils_obj.dir/gpu_common.cpp.o.requires

CMakeFiles/utils_obj.dir/gpu_common.cpp.o.provides: CMakeFiles/utils_obj.dir/gpu_common.cpp.o.requires
	$(MAKE) -f CMakeFiles/utils_obj.dir/build.make CMakeFiles/utils_obj.dir/gpu_common.cpp.o.provides.build
.PHONY : CMakeFiles/utils_obj.dir/gpu_common.cpp.o.provides

CMakeFiles/utils_obj.dir/gpu_common.cpp.o.provides.build: CMakeFiles/utils_obj.dir/gpu_common.cpp.o


CMakeFiles/utils_obj.dir/gpu_utils.cpp.o: CMakeFiles/utils_obj.dir/flags.make
CMakeFiles/utils_obj.dir/gpu_utils.cpp.o: ../gpu_utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/global/homes/m/mgawan/mhm2_kcount/kcount_kernel_driver/gpu-utils/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/utils_obj.dir/gpu_utils.cpp.o"
	/usr/common/software/sles15_cgpu/cuda/11.1.1/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /global/homes/m/mgawan/mhm2_kcount/kcount_kernel_driver/gpu-utils/gpu_utils.cpp -o CMakeFiles/utils_obj.dir/gpu_utils.cpp.o

CMakeFiles/utils_obj.dir/gpu_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/utils_obj.dir/gpu_utils.cpp.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/utils_obj.dir/gpu_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/utils_obj.dir/gpu_utils.cpp.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/utils_obj.dir/gpu_utils.cpp.o.requires:

.PHONY : CMakeFiles/utils_obj.dir/gpu_utils.cpp.o.requires

CMakeFiles/utils_obj.dir/gpu_utils.cpp.o.provides: CMakeFiles/utils_obj.dir/gpu_utils.cpp.o.requires
	$(MAKE) -f CMakeFiles/utils_obj.dir/build.make CMakeFiles/utils_obj.dir/gpu_utils.cpp.o.provides.build
.PHONY : CMakeFiles/utils_obj.dir/gpu_utils.cpp.o.provides

CMakeFiles/utils_obj.dir/gpu_utils.cpp.o.provides.build: CMakeFiles/utils_obj.dir/gpu_utils.cpp.o


utils_obj: CMakeFiles/utils_obj.dir/gpu_common.cpp.o
utils_obj: CMakeFiles/utils_obj.dir/gpu_utils.cpp.o
utils_obj: CMakeFiles/utils_obj.dir/build.make

.PHONY : utils_obj

# Rule to build all files generated by this target.
CMakeFiles/utils_obj.dir/build: utils_obj

.PHONY : CMakeFiles/utils_obj.dir/build

CMakeFiles/utils_obj.dir/requires: CMakeFiles/utils_obj.dir/gpu_common.cpp.o.requires
CMakeFiles/utils_obj.dir/requires: CMakeFiles/utils_obj.dir/gpu_utils.cpp.o.requires

.PHONY : CMakeFiles/utils_obj.dir/requires

CMakeFiles/utils_obj.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/utils_obj.dir/cmake_clean.cmake
.PHONY : CMakeFiles/utils_obj.dir/clean

CMakeFiles/utils_obj.dir/depend:
	cd /global/homes/m/mgawan/mhm2_kcount/kcount_kernel_driver/gpu-utils/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /global/homes/m/mgawan/mhm2_kcount/kcount_kernel_driver/gpu-utils /global/homes/m/mgawan/mhm2_kcount/kcount_kernel_driver/gpu-utils /global/homes/m/mgawan/mhm2_kcount/kcount_kernel_driver/gpu-utils/build /global/homes/m/mgawan/mhm2_kcount/kcount_kernel_driver/gpu-utils/build /global/homes/m/mgawan/mhm2_kcount/kcount_kernel_driver/gpu-utils/build/CMakeFiles/utils_obj.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/utils_obj.dir/depend
