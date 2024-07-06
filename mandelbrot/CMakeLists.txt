cmake_minimum_required(VERSION 3.16)
project(mandelbrot LANGUAGES C)

# Set the C compiler to nvc and the necessary flags for OpenMP GPU support
set(CMAKE_C_COMPILER nvc)
set(CMAKE_C_FLAGS "-mp=gpu -O2" CACHE STRING "NVC OPENMP")

# Find the OpenMP package
find_package(OpenMP REQUIRED)

# Add executables for sequential, CPU, and GPU versions
add_executable(mandelbrot src/mandelbrot.c)
add_executable(mandelbrotCPU src/mandelbrotCPU.c)
add_executable(mandelbrotGPU src/mandelbrotGPU.c)

# Include directories for mandelbrotSequential (no OpenMP)
target_include_directories(mandelbrot PUBLIC include)

# Include directories and link libraries for mandelbrotCPU (with OpenMP)
target_include_directories(mandelbrotCPU PUBLIC include)
target_link_libraries(mandelbrotCPU PUBLIC OpenMP::OpenMP_C)

# Include directories, compile options, and link libraries for mandelbrotGPU (with OpenMP and GPU)
target_include_directories(mandelbrotGPU PUBLIC include)
target_compile_options(mandelbrotGPU PRIVATE -mp=gpu)
target_link_libraries(mandelbrotGPU PUBLIC OpenMP::OpenMP_C)
