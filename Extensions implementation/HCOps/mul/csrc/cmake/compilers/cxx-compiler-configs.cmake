# ==================================================================================================
# @file compiler-configs-cpp.cmake
# @brief Compiler configurations for the host.
#
# @note Several parameters SHOULD be set BEFORE including this file:
#         - `ENV{CXX}`: C++ Compiler. Default: auto-detected.
#         - `CMAKE_CXX_STANDARD`: C++ Standard. Default: 20.
#         - `CMAKE_CXX_SCAN_FOR_MODULES`: Whether to use modules. Default: OFF.
# ==================================================================================================

enable_language(CXX)

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Generate compile_commands.json in build directory
#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")

# Compiler flags for Clang
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    string(APPEND CMAKE_CXX_FLAGS " -fopenmp -Wall -Wextra -Werror")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3")
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g")
# Compiler flags for GNU
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    string(APPEND CMAKE_CXX_FLAGS " -fopenmp -Wall -Wextra -Werror")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3")
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g")
else()
    log_fatal("Unsupported compiler")
endif()
