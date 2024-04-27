#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h> // Library for OpenMP
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

// Structure to represent RGB color
typedef struct {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} RGB;

// Function to calculate the color of a point in the Mandelbrot set
RGB mandelbrot(int x, int y, int WIDTH, int HEIGHT, int MAX_ITER) {
    // Map pixel coordinates to the complex plane
    double creal = (x - WIDTH / 2.0) * 4.0 / WIDTH;
    double cimag = (y - HEIGHT / 2.0) * 4.0 / HEIGHT;
    double real = 0, imag = 0;
    int n = 0;

    // Mandelbrot calculation loop
    while (real * real + imag * imag <= 4 && n < MAX_ITER) {
        double temp = real * real - imag * imag + creal;
        imag = 2 * real * imag + cimag;
        real = temp;
        n++;
    }

    // Assign color based on the iteration count
    RGB color;
    if (n == MAX_ITER) {
        // Black for points inside the set
        color.r = 0;
        color.g = 0;
        color.b = 0;
    } else {
        // Color based on the iteration count
        double t = (double)n / MAX_ITER;
        color.r = (uint8_t)(30 * (1 - t) * t * t * t * 255);
        color.g = (uint8_t)(45 * (1 - t) * (1 - t) * t * t * 255);
        color.b = (uint8_t)(60 * (1 - t) * (1 - t) * (1 - t) * t * 255);
    }
    return color;
}

int main() {
    clock_t start, end;
    double cpu_time_used = 0;

    // Image dimensions and maximum iterations
    int WIDTH = 7680;
    int HEIGHT = 4320;
    int MAX_ITER = 1000;
    double max_time = 600.0; // Maximum execution time
    int images_generated = 0;

    start = clock(); // Mark the start of execution time
    while (cpu_time_used < max_time) {
        // Allocate memory for the image
        RGB *image = (RGB *)malloc(sizeof(RGB) * WIDTH * HEIGHT);
        if (!image) {
            perror("Unable to allocate memory");
            exit(1);
        }

        // Calculate the Mandelbrot image in parallel using OpenMP
        int y, x;
        omp_set_num_threads(25); // Set the number of OpenMP threads
        #pragma omp parallel for private(x, y) shared(image)
        for (y = 0; y < HEIGHT; y++) {
            for (x = 0; x < WIDTH; x++) {
                image[y * WIDTH + x] = mandelbrot(x, y, WIDTH, HEIGHT, MAX_ITER);
            }
        }

        // Save the image to a JPEG file
        stbi_write_jpg("../images/mandelbrotCPU.jpg", WIDTH, HEIGHT, sizeof(RGB), image, 100);
        images_generated++;

        end = clock(); // Mark the end of execution time
        cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC; // Calculate the execution time
        free(image); // Free the memory of the image
    }

    // Print the total execution time and the number of images generated
    printf("Execution time: %f seconds\n", cpu_time_used);
    printf("Images generated: %d\n", images_generated);

    return 0;
}
