# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

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
CMAKE_SOURCE_DIR = /home/zyb/cv/dlib/dlib/examples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zyb/cv/dlib/dlib/examples

# Include any dependencies generated for this target.
include CMakeFiles/threaded_object_ex.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/threaded_object_ex.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/threaded_object_ex.dir/flags.make

CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.o: CMakeFiles/threaded_object_ex.dir/flags.make
CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.o: threaded_object_ex.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/zyb/cv/dlib/dlib/examples/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.o -c /home/zyb/cv/dlib/dlib/examples/threaded_object_ex.cpp

CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/zyb/cv/dlib/dlib/examples/threaded_object_ex.cpp > CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.i

CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/zyb/cv/dlib/dlib/examples/threaded_object_ex.cpp -o CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.s

CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.o.requires:
.PHONY : CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.o.requires

CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.o.provides: CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.o.requires
	$(MAKE) -f CMakeFiles/threaded_object_ex.dir/build.make CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.o.provides.build
.PHONY : CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.o.provides

CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.o.provides.build: CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.o

# Object files for target threaded_object_ex
threaded_object_ex_OBJECTS = \
"CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.o"

# External object files for target threaded_object_ex
threaded_object_ex_EXTERNAL_OBJECTS =

threaded_object_ex: CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.o
threaded_object_ex: CMakeFiles/threaded_object_ex.dir/build.make
threaded_object_ex: dlib_build/libdlib.a
threaded_object_ex: /usr/lib/x86_64-linux-gnu/libnsl.so
threaded_object_ex: /usr/lib/x86_64-linux-gnu/libSM.so
threaded_object_ex: /usr/lib/x86_64-linux-gnu/libICE.so
threaded_object_ex: /usr/lib/x86_64-linux-gnu/libX11.so
threaded_object_ex: /usr/lib/x86_64-linux-gnu/libXext.so
threaded_object_ex: /usr/lib/x86_64-linux-gnu/libpng.so
threaded_object_ex: /usr/lib/x86_64-linux-gnu/libjpeg.so
threaded_object_ex: /usr/lib/libcblas.so
threaded_object_ex: /usr/lib/liblapack.so
threaded_object_ex: CMakeFiles/threaded_object_ex.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable threaded_object_ex"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/threaded_object_ex.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/threaded_object_ex.dir/build: threaded_object_ex
.PHONY : CMakeFiles/threaded_object_ex.dir/build

CMakeFiles/threaded_object_ex.dir/requires: CMakeFiles/threaded_object_ex.dir/threaded_object_ex.cpp.o.requires
.PHONY : CMakeFiles/threaded_object_ex.dir/requires

CMakeFiles/threaded_object_ex.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/threaded_object_ex.dir/cmake_clean.cmake
.PHONY : CMakeFiles/threaded_object_ex.dir/clean

CMakeFiles/threaded_object_ex.dir/depend:
	cd /home/zyb/cv/dlib/dlib/examples && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zyb/cv/dlib/dlib/examples /home/zyb/cv/dlib/dlib/examples /home/zyb/cv/dlib/dlib/examples /home/zyb/cv/dlib/dlib/examples /home/zyb/cv/dlib/dlib/examples/CMakeFiles/threaded_object_ex.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/threaded_object_ex.dir/depend
