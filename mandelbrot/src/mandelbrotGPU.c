#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <omp.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#define WIDTH 15360
#define HEIGHT 8640
#define MAX_ITER 100

void get_color(int iter, int *r, int *g, int *b) {
    if (iter == MAX_ITER) {
        *r = *g = *b = 0;  // Black for points inside the set
    } else {
        double t = (double)iter / MAX_ITER;
        *r = (int)(9 * (1 - t) * t * t * t * 255);
        *g = (int)(15 * (1 - t) * (1 - t) * t * t * 255);
        *b = (int)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
    }
}

void mandelbrot(unsigned char *image) {
    #pragma omp target teams distribute parallel for simd collapse(2)
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double complex c = (4.0 * x / WIDTH - 2.0) + (4.0 * y / HEIGHT - 2.0) * I;
            double complex z = 0;
            int iter = 0;
            while (cabs(z) < 2 && iter < MAX_ITER) {
                z = z * z + c;
                iter++;
            }
            int r, g, b;
            get_color(iter, &r, &g, &b);
            int index = (y * WIDTH + x) * 3;
            image[index + 0] = r;  // Red
            image[index + 1] = g;  // Green
            image[index + 2] = b;  // Blue
        }
    }
}

int main() {
    unsigned char *image = (unsigned char *)malloc(WIDTH * HEIGHT * 3);
    if (image == NULL) {
        fprintf(stderr, "Failed to allocate memory for image\n");
        return 1;
    }

    mandelbrot(image);

    if (!stbi_write_png("mandelbrot.png", WIDTH, HEIGHT, 3, image, WIDTH * 3)) {
        fprintf(stderr, "Failed to write image\n");
        free(image);
        return 1;
    }

    free(image);
    return 0;
}
