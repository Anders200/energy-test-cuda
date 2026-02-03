cmake_minimum_required(VERSION 3.18)
project(energy_test VERSION 0.1 LANGUAGES CXX)

option(ENERGY_TEST_BUILD_SHARED "Build shared library instead of static" ON)
option(ENERGY_TEST_ENABLE_CUDA "Enable CUDA language/toolkit (if available)" OFF)
option(ENERGY_TEST_ENABLE_PYTHON "Build Python module via pybind11" ON)

if(ENERGY_TEST_ENABLE_CUDA)
    enable_language(CUDA)
    add_executable(energy_test_app 
        src/main.cpp 
        src/cuda_energy_test.cu 
        src/binder.cpp
    )
    target_link_libraries(energy_test_app PRIVATE energy_binder)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- Libraries ----

# Header-only (templates) but keep as an INTERFACE dependency to model includes.
add_library(cpu_algorithms INTERFACE)
target_include_directories(cpu_algorithms INTERFACE ${PROJECT_SOURCE_DIR}/include)

# Binder library (the stable API we will later extend with CUDA backends)
if(ENERGY_TEST_BUILD_SHARED)
    add_library(energy_binder SHARED src/binder.cpp)
else()
    add_library(energy_binder STATIC src/binder.cpp)
endif()
target_include_directories(energy_binder PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(energy_binder PUBLIC cpu_algorithms)

# ---- Python bindings (pybind11) ----
if(ENERGY_TEST_ENABLE_PYTHON)
    # Prefer modern FindPython to avoid ABI mismatch (python3.11 vs python3.12 etc.).
    # In the Nix devShell you provided, Python is 3.11 with pybind11 installed.
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(energy_py src/python/energy_py.cpp)
    target_link_libraries(energy_py PRIVATE energy_binder)
    target_include_directories(energy_py PRIVATE ${PROJECT_SOURCE_DIR}/include)
endif()
