#define MANDELBROTGPU_H

#include <cuda_runtime.h>


__device__ RGB mandelbrot(int x, int y, int WIDTH, int HEIGHT, int MAX_ITER);

__global__ void mandelbrot_kernel(RGB *image);