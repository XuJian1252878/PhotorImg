# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/Applications/CLion.app/Contents/bin/cmake/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/Applications/CLion.app/Contents/bin/cmake/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/CMakeFiles /Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /Users/xujian/Workspace/AndroidStudy/CPlusPlus/ImageRegistration/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named ImageRegistration

# Build rule for target.
ImageRegistration: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 ImageRegistration
.PHONY : ImageRegistration

# fast build rule for target.
ImageRegistration/fast:
	$(MAKE) -f CMakeFiles/ImageRegistration.dir/build.make CMakeFiles/ImageRegistration.dir/build
.PHONY : ImageRegistration/fast

StarImage.o: StarImage.cpp.o

.PHONY : StarImage.o

# target to build an object file
StarImage.cpp.o:
	$(MAKE) -f CMakeFiles/ImageRegistration.dir/build.make CMakeFiles/ImageRegistration.dir/StarImage.cpp.o
.PHONY : StarImage.cpp.o

StarImage.i: StarImage.cpp.i

.PHONY : StarImage.i

# target to preprocess a source file
StarImage.cpp.i:
	$(MAKE) -f CMakeFiles/ImageRegistration.dir/build.make CMakeFiles/ImageRegistration.dir/StarImage.cpp.i
.PHONY : StarImage.cpp.i

StarImage.s: StarImage.cpp.s

.PHONY : StarImage.s

# target to generate assembly for a file
StarImage.cpp.s:
	$(MAKE) -f CMakeFiles/ImageRegistration.dir/build.make CMakeFiles/ImageRegistration.dir/StarImage.cpp.s
.PHONY : StarImage.cpp.s

StarImagePart.o: StarImagePart.cpp.o

.PHONY : StarImagePart.o

# target to build an object file
StarImagePart.cpp.o:
	$(MAKE) -f CMakeFiles/ImageRegistration.dir/build.make CMakeFiles/ImageRegistration.dir/StarImagePart.cpp.o
.PHONY : StarImagePart.cpp.o

StarImagePart.i: StarImagePart.cpp.i

.PHONY : StarImagePart.i

# target to preprocess a source file
StarImagePart.cpp.i:
	$(MAKE) -f CMakeFiles/ImageRegistration.dir/build.make CMakeFiles/ImageRegistration.dir/StarImagePart.cpp.i
.PHONY : StarImagePart.cpp.i

StarImagePart.s: StarImagePart.cpp.s

.PHONY : StarImagePart.s

# target to generate assembly for a file
StarImagePart.cpp.s:
	$(MAKE) -f CMakeFiles/ImageRegistration.dir/build.make CMakeFiles/ImageRegistration.dir/StarImagePart.cpp.s
.PHONY : StarImagePart.cpp.s

StarImageRegistBuilder.o: StarImageRegistBuilder.cpp.o

.PHONY : StarImageRegistBuilder.o

# target to build an object file
StarImageRegistBuilder.cpp.o:
	$(MAKE) -f CMakeFiles/ImageRegistration.dir/build.make CMakeFiles/ImageRegistration.dir/StarImageRegistBuilder.cpp.o
.PHONY : StarImageRegistBuilder.cpp.o

StarImageRegistBuilder.i: StarImageRegistBuilder.cpp.i

.PHONY : StarImageRegistBuilder.i

# target to preprocess a source file
StarImageRegistBuilder.cpp.i:
	$(MAKE) -f CMakeFiles/ImageRegistration.dir/build.make CMakeFiles/ImageRegistration.dir/StarImageRegistBuilder.cpp.i
.PHONY : StarImageRegistBuilder.cpp.i

StarImageRegistBuilder.s: StarImageRegistBuilder.cpp.s

.PHONY : StarImageRegistBuilder.s

# target to generate assembly for a file
StarImageRegistBuilder.cpp.s:
	$(MAKE) -f CMakeFiles/ImageRegistration.dir/build.make CMakeFiles/ImageRegistration.dir/StarImageRegistBuilder.cpp.s
.PHONY : StarImageRegistBuilder.cpp.s

Util.o: Util.cpp.o

.PHONY : Util.o

# target to build an object file
Util.cpp.o:
	$(MAKE) -f CMakeFiles/ImageRegistration.dir/build.make CMakeFiles/ImageRegistration.dir/Util.cpp.o
.PHONY : Util.cpp.o

Util.i: Util.cpp.i

.PHONY : Util.i

# target to preprocess a source file
Util.cpp.i:
	$(MAKE) -f CMakeFiles/ImageRegistration.dir/build.make CMakeFiles/ImageRegistration.dir/Util.cpp.i
.PHONY : Util.cpp.i

Util.s: Util.cpp.s

.PHONY : Util.s

# target to generate assembly for a file
Util.cpp.s:
	$(MAKE) -f CMakeFiles/ImageRegistration.dir/build.make CMakeFiles/ImageRegistration.dir/Util.cpp.s
.PHONY : Util.cpp.s

main.o: main.cpp.o

.PHONY : main.o

# target to build an object file
main.cpp.o:
	$(MAKE) -f CMakeFiles/ImageRegistration.dir/build.make CMakeFiles/ImageRegistration.dir/main.cpp.o
.PHONY : main.cpp.o

main.i: main.cpp.i

.PHONY : main.i

# target to preprocess a source file
main.cpp.i:
	$(MAKE) -f CMakeFiles/ImageRegistration.dir/build.make CMakeFiles/ImageRegistration.dir/main.cpp.i
.PHONY : main.cpp.i

main.s: main.cpp.s

.PHONY : main.s

# target to generate assembly for a file
main.cpp.s:
	$(MAKE) -f CMakeFiles/ImageRegistration.dir/build.make CMakeFiles/ImageRegistration.dir/main.cpp.s
.PHONY : main.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... edit_cache"
	@echo "... ImageRegistration"
	@echo "... StarImage.o"
	@echo "... StarImage.i"
	@echo "... StarImage.s"
	@echo "... StarImagePart.o"
	@echo "... StarImagePart.i"
	@echo "... StarImagePart.s"
	@echo "... StarImageRegistBuilder.o"
	@echo "... StarImageRegistBuilder.i"
	@echo "... StarImageRegistBuilder.s"
	@echo "... Util.o"
	@echo "... Util.i"
	@echo "... Util.s"
	@echo "... main.o"
	@echo "... main.i"
	@echo "... main.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

