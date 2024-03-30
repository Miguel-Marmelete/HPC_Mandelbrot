cmake_minimum_required(VERSION 3.8)
project(mandelbrot CUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_30)


add_executable(mandelbrot src/main.c )



target_include_directories(mandelbrot PUBLIC include)
target_sources(mandelbrot PUBLIC include)

