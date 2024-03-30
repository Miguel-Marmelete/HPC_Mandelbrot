#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <rgb.h>
#include <mandelbrotCPU.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#define WIDTH 7680
#define HEIGHT 4230
#define MAX_ITER 500


int main() {
    clock_t start, end;
    double cpu_time_used;
    start = clock();

    RGB *image = (RGB *)malloc(sizeof(RGB) * WIDTH * HEIGHT);
    if (!image) {
        perror("Unable to allocate memory");
        exit(1);
    }
    
    #pragma omp parallel for collapse(5)
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            image[y * WIDTH + x] = mandelbrot(x, y, WIDTH, HEIGHT , MAX_ITER);
        }
    }

    stbi_write_jpg("../images/mandelbrot.jpg", WIDTH, HEIGHT, sizeof(RGB), image, 100); // Quality set to 100

    free(image);
    end = clock(); // Marca o tempo final

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Tempo de execucao: %f segundos\n", cpu_time_used);
    return 0;
}
