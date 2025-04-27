# Minimum required CMake version
cmake_minimum_required(VERSION 3.22)

# Project Name and version
project(TORCHOP VERSION 1.0.0)

set(PROJECT_NAMESPACE "kpops::")

# Utility functions for installing libraries:
include(${PROJECT_SOURCE_DIR}/cmake/utils/lib-install.cmake)

# Compiler configurations for cpp, e.g., g++, clang++, etc.
include(${PROJECT_SOURCE_DIR}/cmake/compilers/cxx-compiler-configs.cmake)

include(${PROJECT_SOURCE_DIR}/cmake/libraries/libpython.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/libraries/libtorch.cmake)

add_subdirectory(${PROJECT_SOURCE_DIR}/lib)
