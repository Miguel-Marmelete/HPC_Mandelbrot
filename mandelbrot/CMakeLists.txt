cmake_minimum_required(VERSION 3.8)
project(mandelbrot LANGUAGES CUDA C)

# Find CUDA
find_package(CUDA REQUIRED)

find_package(OpenMP)

# Include directories for CUDA
include_directories(${CUDA_INCLUDE_DIRS})

# Executable for mandelbrotCPU.c
add_executable(mandelbrotCPU src/mandelbrotCPU.c)

add_executable(mandelbrotSequential src/mandelbrot.c)

# Executable for mandelbrotGPU.cu
cuda_add_executable(mandelbrotGPU src/mandelbrotGPU.cu)

# Link CUDA library to mandelbrot2
#target_link_libraries(mandelbrot2 PRIVATE ${CUDA_LIBRARIES})

# Link OpenMP to both executables if found
target_link_libraries(mandelbrotCPU PUBLIC OpenMP::OpenMP_C)
#target_link_libraries(mandelbrot2 PUBLIC OpenMP::OpenMP_CXX)

# Include directories for mandelbrot
target_include_directories(mandelbrotSequential PUBLIC include)

target_include_directories(mandelbrotCPU PUBLIC include)

# Include directories for mandelbrot2
target_include_directories(mandelbrotGPU PUBLIC include)
