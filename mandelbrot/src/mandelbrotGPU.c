#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define WIDTH 7680
#define HEIGHT 4320

typedef struct {
    uint8_t r, g, b;
} pixel_t;

typedef struct {
    int width, height;
    pixel_t *data;
} Image;

const pixel_t pixel_colour[16] = {
    {66, 30, 15}, {25, 7, 26}, {9, 1, 47}, {4, 4, 73},
    {0, 7, 100}, {12, 44, 138}, {24, 82, 177}, {57, 125, 209},
    {134, 181, 229}, {211, 236, 248}, {241, 233, 191}, {248, 201, 95},
    {255, 170, 0}, {204, 128, 0}, {153, 87, 0}, {106, 52, 3}
};

void generate_mandelbrot(Image *image, double cx, double cy) {
    uint8_t iter_max = 100;
    double scale = 1.0 / (image->width / 4.0);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < image->height; i++) {
        for (int j = 0; j < image->width; j++) {
            const double y = (i - image->height / 2) * scale + cy;
            const double x = (j - image->width / 2) * scale + cx;
            double zx, zy, zx2, zy2;
            uint8_t iter = 0;
            zx = hypot(x - 0.25, y);

            if (x < zx - 2 * zx * zx + 0.25 || (x + 1) * (x + 1) + y * y < 0.0625) iter = iter_max;

            zx = zy = zx2 = zy2 = 0;

            do {
                zy = 2.0 * zx * zy + y;
                zx = zx2 - zy2 + x;
                zx2 = zx * zx;
                zy2 = zy * zy;
            } while (iter++ < iter_max && zx2 + zy2 < 4.0);

            if (iter > 0 && iter < iter_max) {
                const uint8_t idx = iter % 16;
                image->data[i * image->width + j] = pixel_colour[idx];
            }
        }
    }
}

void save_image_jpeg(const char *filename, Image *image) {
    uint8_t *rgb_image = (uint8_t *)malloc(image->width * image->height * 3 * sizeof(uint8_t));
    if (!rgb_image) {
        perror("Failed to allocate memory");
        exit(EXIT_FAILURE);
    }

    // Convert pixel_t data to RGB format (3 bytes per pixel)
    for (int i = 0; i < image->width * image->height; ++i) {
        rgb_image[i * 3] = image->data[i].r;
        rgb_image[i * 3 + 1] = image->data[i].g;
        rgb_image[i * 3 + 2] = image->data[i].b;
    }

    // Save RGB image as JPEG using stb_image_write.h
    if (!stbi_write_jpg(filename, image->width, image->height, 3, rgb_image, 100)) {
        perror("Failed to write JPEG file");
        exit(EXIT_FAILURE);
    }

    free(rgb_image);
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <time_limit_seconds>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int time_limit = atoi(argv[1]);
    if (time_limit <= 0) {
        fprintf(stderr, "Invalid time limit: %s\n", argv[1]);
        return EXIT_FAILURE;
    }

    Image result;
    result.width = WIDTH;
    result.height = HEIGHT;
    result.data = (pixel_t *)malloc(result.width * result.height * sizeof(pixel_t));
    if (!result.data) {
        perror("Failed to allocate memory");
        return EXIT_FAILURE;
    }

    double cx = -0.6, cy = 0.0;
    int images_generated = 0;
    double start_time = omp_get_wtime();

    do {
        generate_mandelbrot(&result, cx, cy);
        images_generated++;

        double current_time = omp_get_wtime();
        if (current_time - start_time >= time_limit) {
            break;
        }
    } while (1);

    double end_time = omp_get_wtime();
    printf("Time taken: %.4f seconds.\n", end_time - start_time);
    printf("Number of images generated: %d\n", images_generated);

    char filename[50];
    snprintf(filename, sizeof(filename), "../images/mandelbrotCPU.jpg");
    save_image_jpeg(filename, &result);

    free(result.data);
    return EXIT_SUCCESS;
}
