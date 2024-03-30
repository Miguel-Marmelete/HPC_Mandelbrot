cmake_minimum_required(VERSION 3.10)
project(mandelbrot C)

add_executable(mandelbrot src/main.c)


target_include_directories(mandelbrot PUBLIC include)
target_sources(mandelbrot PUBLIC include)