cmake_minimum_required(VERSION 3.18)

project(energy_test
    VERSION 0.1
    LANGUAGES CXX
)

# ------------------------
# Global configuration
# ------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Compiler warnings
if (MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ------------------------
# Library: cpu_baseline
# ------------------------
add_library(cpu_baseline
    src/cpu_baseline.cpp
)

target_include_directories(cpu_baseline
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

# ------------------------
# Executable
# ------------------------
add_executable(energy_test_app
    src/main.cpp
)

target_link_libraries(energy_test_app
    PRIVATE cpu_baseline
)

# ------------------------
# Future extensions
# ------------------------

# ---- CUDA (later) ----
# enable_language(CUDA)
# set(CMAKE_CUDA_STANDARD 17)
# add_library(cuda_backend src/cuda_energy.cu)
# target_link_libraries(cpu_baseline PUBLIC cuda_backend)

# ---- CPython / pybind11 (later) ----
# find_package(Python COMPONENTS Development REQUIRED)
# target_link_libraries(cpu_baseline PUBLIC Python::Python)
#
# OR (preferred)
# find_package(pybind11 REQUIRED)
# pybind11_add_module(energy_py src/python/energy_py.cpp)
# target_link_libraries(energy_py PRIVATE cpu_baseline)

