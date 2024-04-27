#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <cuda_runtime.h>

// Image dimensions
#define WIDTH 7680
#define HEIGHT 4320

// Maximum iterations for Mandelbrot calculation
#define MAX_ITER 1000


// Struct to represent RGB color
typedef struct {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} RGB;

// Device function to calculate Mandelbrot color for a pixel
__device__ RGB mandelbrot(int x, int y) {
    double creal = (x - WIDTH / 2.0) * 4.0 / WIDTH;
    double cimag = (y - HEIGHT / 2.0) * 4.0 / HEIGHT;
    double real = 0, imag = 0;
    int n = 0;

// Mandelbrot calculation loop
    while (real * real + imag * imag <= 20 && n < MAX_ITER) {
        double temp = real * real - imag * imag + creal;
        imag = 2 * real * imag + cimag;
        real = temp;
        n++;
    }

    RGB color;
    // Assign color based on iteration count
    if (n == MAX_ITER) {
        color.r = 0;
        color.g = 0;
        color.b = 0;
    } else {
        double t = (double)n / MAX_ITER;
        color.r = (uint8_t)(30 * (1 - t) * t * t * t * 255);
        color.g = (uint8_t)(45* (1 - t) * (1 - t) * t * t * 255);
        color.b = (uint8_t)(60 * (1 - t) * (1 - t) * (1 - t) * t * 255);
    }
    return color;
}
// Kernel function to compute Mandelbrot colors for each pixel
__global__ void mandelbrot_kernel(RGB *image) {
    // Thread indices
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < WIDTH && y < HEIGHT) {
        // Calculate Mandelbrot color for the pixel
        image[y * WIDTH + x] = mandelbrot(x, y);
    }
}

int main() {
    clock_t start, end;
    double gpu_time_used =0;
    double max_time = 3600.0;
    int images_generated = 0;
    start = clock();
    while(gpu_time_used < max_time){
        // Allocate memory for the image on CPU
        RGB *image = (RGB *)malloc(sizeof(RGB) * WIDTH * HEIGHT);
        // Declare GPU memory pointer
        RGB *d_image;

        // Allocate memory on GPU
        cudaMalloc((void **)&d_image, sizeof(RGB) * WIDTH * HEIGHT);

        // Define grid and block dimensions for GPU parallelism
        dim3 threadsPerBlock(5, 5); // 16x16 threads per block
        dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y); // Suficiente para cobrir toda a imagem

        // Launch Mandelbrot kernel on GPU
        mandelbrot_kernel<<<numBlocks, threadsPerBlock>>>(d_image);

        // Copy the result back to CPU
        cudaMemcpy(image, d_image, sizeof(RGB) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
        stbi_write_jpg("../images/mandelbrotGPU.jpg", WIDTH, HEIGHT, sizeof(RGB), image, 100);
        images_generated++;
        end = clock(); // Marca o tempo final
        gpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

        cudaFree(d_image);
        free(image);

        
    }
    printf("Tempo de execucao: %f segundos\n", gpu_time_used);
    printf("Images generated: %d\n", images_generated);

    return 0;
}
