#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>



typedef struct {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} RGB;

RGB mandelbrot(int x, int y, int WIDTH, int HEIGHT,int MAX_ITER ) {
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

int main() {
    int WIDTH = 7680;
    int HEIGHT = 4320;
    int MAX_ITER = 1000;
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    RGB *image = (RGB *)malloc(sizeof(RGB) * WIDTH * HEIGHT);
    if (!image) {
        perror("Unable to allocate memory");
        exit(1);
    }
    
    int y;
    int x; 
    omp_set_num_threads(10);
    #pragma omp parallel for private(x,y) shared(image) 
    for (y = 0; y < HEIGHT; y++) {
        for (x = 0; x < WIDTH; x++) {
            image[y * WIDTH + x] = mandelbrot(x, y,WIDTH, HEIGHT, MAX_ITER);
        }
    }
 

    stbi_write_jpg("../images/mandelbrotCPU.jpg", WIDTH, HEIGHT, sizeof(RGB), image, 100);

    free(image);
    end = clock(); // Mark the end time

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Execution time: %f seconds\n", cpu_time_used);
    return 0;
}
