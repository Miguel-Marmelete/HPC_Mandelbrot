#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <omp.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"

#define WIDTH 7680
#define HEIGHT 4320
#define MAX_ITER 1000

void mandelbrot(unsigned char *image) {
    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double complex c = (4.0 * x / WIDTH - 2.0) + (4.0 * y / HEIGHT - 2.0) * I;
            double complex z = 0;
            int iter = 0;
            while (cabs(z) < 2 && iter < MAX_ITER) {
                z = z * z + c;
                iter++;
            }
            int color = (iter == MAX_ITER) ? 0 : (255 * iter / MAX_ITER);
            int index = (y * WIDTH + x) * 3;
            image[index + 0] = color;  // Red
            image[index + 1] = color;  // Green
            image[index + 2] = color;  // Blue
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
