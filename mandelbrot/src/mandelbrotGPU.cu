#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <cuda_runtime.h>

#define WIDTH 7680
#define HEIGHT 4320
#define MAX_ITER 10000

typedef struct {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} RGB;

__device__ RGB mandelbrot(int x, int y) {
    double creal = (x - WIDTH / 2.0) * 4.0 / WIDTH;
    double cimag = (y - HEIGHT / 2.0) * 4.0 / HEIGHT;
    double real = 0, imag = 0;
    int n = 0;

    while (real * real + imag * imag <= 4 && n < MAX_ITER) {
        double temp = real * real - imag * imag + creal;
        imag = 2 * real * imag + cimag;
        real = temp;
        n++;
    }

    RGB color;
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

__global__ void mandelbrot_kernel(RGB *image) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //omp_set_num_threads(10);
    //#pragma omp parallel for 
    if (x < WIDTH && y < HEIGHT) {
        image[y * WIDTH + x] = mandelbrot(x, y);
    }
}

int main() {
    clock_t start, end;
    double cpu_time_used;
    start = clock();

    RGB *image = (RGB *)malloc(sizeof(RGB) * WIDTH * HEIGHT);
    RGB *d_image;

    cudaMalloc((void **)&d_image, sizeof(RGB) * WIDTH * HEIGHT);

    dim3 threadsPerBlock(16, 16); // 16x16 threads per block
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y); // Suficiente para cobrir toda a imagem

    mandelbrot_kernel<<<numBlocks, threadsPerBlock>>>(d_image);

    cudaMemcpy(image, d_image, sizeof(RGB) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
    stbi_write_jpg("../images/mandelbrotGPU.jpg", WIDTH, HEIGHT, sizeof(RGB), image, 100);

    cudaFree(d_image);
    free(image);

    end = clock(); // Marca o tempo final
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Tempo de execucao: %f segundos\n", cpu_time_used);

    return 0;
}
