cmake_minimum_required(VERSION 3.8)
project(mandelbrot)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Executable for mandelbrotCPU.c
add_executable(mandelbrot src/mandelbrotCPU.c)

# Executable for mandelbrotGPU.cu
cuda_add_executable(mandelbrot2 src/mandelbrotGPU.cu)

# Include directories for mandelbrotCPU
target_include_directories(mandelbrot PUBLIC include)

# Include directories for mandelbrotGPU
target_include_directories(mandelbrot2 PUBLIC include)



target_include_directories(mandelbrot PUBLIC include)
target_sources(mandelbrot PUBLIC include)

