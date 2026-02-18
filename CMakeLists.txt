cmake_minimum_required(VERSION 3.18)
project(energy_test VERSION 0.1 LANGUAGES CXX)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(ENERGY_TEST_BUILD_SHARED "Build shared library instead of static" ON)
option(ENERGY_TEST_ENABLE_CUDA "Enable CUDA language/toolkit (if available)" OFF)
option(ENERGY_TEST_ENABLE_PYTHON "Build Python module via pybind11" ON)

if(ENERGY_TEST_ENABLE_CUDA)
    # Only enable CUDA when requested. This avoids hard-failing on machines
    # without CUDA headers/toolkit.
    # Default to 'native' if the user didn't specify an architecture
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    enable_language(CUDA)
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

# ---- CUDA backend (optional) ----
if(ENERGY_TEST_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

    add_library(cuda_backend STATIC
        src/kernels.cu
        src/energy_test.cpp
    )
    set_target_properties(cuda_backend PROPERTIES CUDA_STANDARD 17)
    set_target_properties(cuda_backend PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    target_include_directories(cuda_backend PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(cuda_backend PUBLIC CUDA::cudart)

    target_link_libraries(energy_binder PUBLIC cuda_backend)
    target_compile_definitions(energy_binder PUBLIC ENERGY_TEST_HAS_CUDA=1)
endif()

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
