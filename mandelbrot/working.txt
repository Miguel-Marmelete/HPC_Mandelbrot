cmake_minimum_required(VERSION 3.8)
project(mandelbrot LANGUAGES CUDA C)

# Find CUDA
find_package(CUDA REQUIRED)

find_package(OpenMP)

# Include directories for CUDA
include_directories(${CUDA_INCLUDE_DIRS})

# Executable for mandelbrotCPU.c
add_executable(mandelbrot src/mandelbrotCPU.c)

# Executable for mandelbrotGPU.cu
cuda_add_executable(mandelbrot2 src/mandelbrotGPU.cu)

# Link CUDA library to mandelbrot2
#target_link_libraries(mandelbrot2 PRIVATE ${CUDA_LIBRARIES})

# Link OpenMP to both executables if found
target_link_libraries(mandelbrot PUBLIC OpenMP::OpenMP_C)
#target_link_libraries(mandelbrot2 PUBLIC OpenMP::OpenMP_CXX)

# Include directories for mandelbrot
target_include_directories(mandelbrot PUBLIC include)

# Include directories for mandelbrot2
target_include_directories(mandelbrot2 PUBLIC include)
