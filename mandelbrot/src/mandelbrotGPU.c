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

void generate_mandelbrot(Image *image, double cx, double cy, int thread_id, int num_threads) {
    int iter_max = 80; // Change from uint8_t to int
    double scale = 1.0 / (image->width / 4.0);

    // Calculate the range of rows each thread will process
    int chunk_size = image->height / num_threads;
    int start_row = thread_id * chunk_size;
    int end_row = (thread_id == num_threads - 1) ? image->height : start_row + chunk_size;

    #pragma omp target teams distribute parallel for
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < image->width; j++) {
            const double y = (i - image->height / 2) * scale + cy;
            const double x = (j - image->width / 2) * scale + cx;
            double zx, zy, zx2, zy2;
            int iter = 0;
            zx = zy = zx2 = zy2 = 0;

            while (iter < iter_max && zx2 + zy2 < 4.0) {
                zy = 2.0 * zx * zy + y;
                zx = zx2 - zy2 + x;
                zx2 = zx * zx;
                zy2 = zy * zy;
                iter++;
            }

            if (iter < iter_max) {
                int color_index = (iter % 16);
                image->data[i * image->width + j] = pixel_colour[color_index];
            } else {
                image->data[i * image->width + j] = (pixel_t){0, 0, 0};
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

    #pragma omp target teams distribute parallel for
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
  

    int num_threads = atoi(argv[1]);
   

    Image result;
    result.width = WIDTH;
    result.height = HEIGHT;
    result.data = (pixel_t *)malloc(result.width * result.height * sizeof(pixel_t));
   

    // Set the number of threads for OpenMP
    omp_set_num_threads(num_threads);

    double cx = -0.6, cy = 0.0;

    // Each thread generates its part of the image
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        generate_mandelbrot(&result, cx, cy, thread_id, num_threads);
    }

    char filename[50];
    snprintf(filename, sizeof(filename), "../images/mandelbrotGPU.jpg");
    save_image_jpeg(filename, &result);

    free(result.data);
    return EXIT_SUCCESS;
}
