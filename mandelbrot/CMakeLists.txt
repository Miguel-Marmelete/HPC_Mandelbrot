cmake_minimum_required(VERSION 3.16)
project(mandelbrot LANGUAGES C)

# Set the C compiler to nvc and the necessary flags for OpenMP GPU support
set(CMAKE_C_COMPILER nvc)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mp=gpu -O2" CACHE STRING "NVC OPENMP")

# Find MPI package
find_package(MPI REQUIRED)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Add executables for sequential, CPU, and GPU versions
add_executable(mandelbrot src/mandelbrot.c)
add_executable(mandelbrotCPU src/mandelbrotCPU.c)
add_executable(mandelbrotGPU src/mandelbrotGPU.c)

# Include directories for mandelbrot (sequential version, no OpenMP)
target_include_directories(mandelbrot PUBLIC include)

# Include directories, compile options, and link libraries for mandelbrotCPU (CPU version with OpenMP)
target_include_directories(mandelbrotCPU PUBLIC include)
target_link_libraries(mandelbrotCPU PUBLIC OpenMP::OpenMP_C)

# Include directories, compile options, and link libraries for mandelbrotGPU (GPU version with OpenMP and MPI)
target_include_directories(mandelbrotGPU PUBLIC include ${MPI_C_INCLUDE_PATH})
target_compile_options(mandelbrotGPU PRIVATE -mp=gpu)
target_link_libraries(mandelbrotGPU PUBLIC OpenMP::OpenMP_C ${MPI_C_LIBRARIES})

# Optional: You can also add custom commands or targets for running the GPU version
# For example, if you want to run the GPU version with specific options
add_custom_target(run_mandelbrotGPU
    COMMAND mpiexec -n 4 ./mandelbrotGPU 2000 1000  # Example command for running with MPI
    DEPENDS mandelbrotGPU
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)
