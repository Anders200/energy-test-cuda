cmake_minimum_required(VERSION 3.18)
project(energy_test VERSION 0.1 LANGUAGES CXX)

# ============================================================
# Options
# ============================================================

option(ENERGY_TEST_BUILD_SHARED "Build shared library instead of static" ON)
option(ENERGY_TEST_ENABLE_CUDA "Enable CUDA backend" ON)
option(ENERGY_TEST_ENABLE_PYTHON "Build Python module via pybind11" ON)

# ============================================================
# C++ standard
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ============================================================
# Output directories
# ============================================================

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ============================================================
# Warnings
# ============================================================

if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ============================================================
# Header-only CPU algorithms
# ============================================================

add_library(cpu_algorithms INTERFACE)
target_include_directories(cpu_algorithms INTERFACE ${PROJECT_SOURCE_DIR}/include)

# ============================================================
# CUDA backend (SEPARATE LIBRARY!)
# ============================================================

if(ENERGY_TEST_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(energy_cuda STATIC
        src/cuda/energy_cuda.cu
        src/cuda/kernels.cu
    )

    target_include_directories(energy_cuda
        PUBLIC ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(energy_cuda
        PUBLIC CUDA::cudart
    )

    set_target_properties(energy_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# ============================================================
# Binder library (CORE API)
# ============================================================

if(ENERGY_TEST_BUILD_SHARED)
    add_library(energy_binder SHARED src/binder.cpp)
else()
    add_library(energy_binder STATIC src/binder.cpp)
endif()

target_include_directories(energy_binder
    PUBLIC ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(energy_binder
    PUBLIC cpu_algorithms
)

if(ENERGY_TEST_ENABLE_CUDA)
    target_link_libraries(energy_binder PUBLIC energy_cuda)
endif()

# ============================================================
# CLI application
# ============================================================

add_executable(energy_test_app src/main.cpp)
target_link_libraries(energy_test_app PRIVATE energy_binder)

# ============================================================
# Python bindings (pybind11)
# ============================================================

if(ENERGY_TEST_ENABLE_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(energy_py
        src/python/energy_py.cpp
    )

    target_include_directories(energy_py
        PRIVATE ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(energy_py
        PRIVATE energy_binder
    )
endif()
