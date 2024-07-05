#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
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

    clock_t start, end;
    double time_used = 0;

    int WIDTH = 7680;
    int HEIGHT = 4320;
    int MAX_ITER = 100;
    double max_time = 60.0;
    int images_generated = 0;

    start = clock();
    while(time_used < max_time){
        RGB *image = (RGB *)malloc(sizeof(RGB) * WIDTH * HEIGHT);
        if (!image) {
            perror("Unable to allocate memory");
            exit(1);
        }
        
        int y;
        int x; 
        for (y = 0; y < HEIGHT; y++) {
            for (x = 0; x < WIDTH; x++) {
                image[y * WIDTH + x] = mandelbrot(x, y,WIDTH, HEIGHT, MAX_ITER);
            }
        }
 

        stbi_write_jpg("../images/mandelbrotCPU.jpg", WIDTH, HEIGHT, sizeof(RGB), image, 100);
        images_generated++;
        end = clock(); // Mark the end time
        time_used = (double)(end - start) / CLOCKS_PER_SEC;

        free(image);
        
    }   
    
    printf("Execution time: %f seconds\n", time_used);
    printf("Images generated: %d\n", images_generated);
    return 0;
}
