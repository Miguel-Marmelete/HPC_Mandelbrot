cmake_minimum_required(VERSION 3.16)
project(mandelbrot LANGUAGES C)
set(CMAKE_C_COMPILER nvc)
set(CMAKE_C_FLAGS "-mp=gpu -O2" CACHE STRING "NVC OPENMP")

find_package(OpenMP)

# Executáveis para mandelbrotSequential (sem OpenMP) e mandelbrotCPU (com OpenMP)
add_executable(mandelbrotSequential src/mandelbrot.c)
add_executable(mandelbrotCPU src/mandelbrotCPU.c)
add_executable(mandelbrotGPU src/mandelbrotGPU.c)
# Configurações para o mandelbrotSequential (sem OpenMP)
target_include_directories(mandelbrotSequential PUBLIC include)

# Configurações para o mandelbrotCPU (com OpenMP, se encontrado)
if(OpenMP_C_FOUND)
    target_link_libraries(mandelbrotCPU PUBLIC OpenMP::OpenMP_C)
    target_include_directories(mandelbrotCPU PUBLIC include)
endif()

if(OpenMP_C_FOUND)
    target_link_libraries(mandelbrotGPU PUBLIC OpenMP::OpenMP_C)
    target_include_directories(mandelbrotGPU PUBLIC include)
    target_compile_options(mandelbrotGPU PRIVATE -mp=gpu)
endif()



#target_link_libraries(mandelbrotCPU PUBLIC OpenMP::OpenMP_C)


# Include directories for mandelbrot
#target_include_directories(mandelbrotSequential PUBLIC include)

#target_include_directories(mandelbrotCPU PUBLIC include)
