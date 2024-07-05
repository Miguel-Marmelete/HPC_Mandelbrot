#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define X 7680
#define Y 4320
#define uchar unsigned char

#define R_MAX 1.5
#define R_MIN -2
#define I_MAX 1.0
#define I_MIN -I_MAX

#define MAX_ITER 1000

typedef struct {
    uchar r;
    uchar g;
    uchar b;
} Color;

double lerp(double v0, double v1, double t) {
    return (1 - t) * v0 + t * v1;
}

Color* make_palette(int size);

Color mandelbrot(int px, int py, Color* palette) {
    double x = 0; // complex (c)
    double y = 0;

    double x0 = R_MIN + (px * ((R_MAX - R_MIN) / (X * 1.0))); // complex scale of Px
    double y0 = I_MIN + (py * ((I_MAX - I_MIN) / (Y * 1.0))); // complex scale of Py

    double i = 0;
    double x2 = 0;
    double y2 = 0;

    while (x2 + y2 <= 20 && i < MAX_ITER) {
        y = 2 * x * y + y0;
        x = x2 - y2 + x0;
        x2 = x * x;
        y2 = y * y;
        i++;
    }
    if (i < MAX_ITER) {
        double log_zn = log(x * x + y * y) / 2.0;
        double nu = log(log_zn / log(2.0)) / log(2.0);
        i += 1.0 - nu;
    }
    Color c1 = palette[(int)i];
    Color c2;
    if ((int)i + 1 > MAX_ITER) {
        c2 = palette[(int)i];
    } else {
        c2 = palette[((int)i) + 1];
    }

    double mod = i - ((int)i); // cant mod doubles
    return (Color){
        .r = (int)lerp(c1.r, c2.r, mod),
        .g = (int)lerp(c1.g, c2.g, mod),
        .b = (int)lerp(c1.b, c2.b, mod),
    };
}

void generate_mandelbrot_image(uchar (*colors)[X][3], Color* palette) {
    #pragma omp parallel for
    for (int Py = 0; Py < Y; Py++) {
        for (int Px = 0; Px < X; Px++) {
            Color c = mandelbrot(Px, Py, palette);
            colors[Py][Px][0] = c.r;
            colors[Py][Px][1] = c.g;
            colors[Py][Px][2] = c.b;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <time_limit_seconds>\n", argv[0]);
        return 1;
    }
    double time_limit = atof(argv[1]);

    uchar (*colors)[X][3] = malloc(sizeof(uchar[Y][X][3]));
    Color* palette = make_palette(MAX_ITER);

    double start_time = omp_get_wtime();
    double current_time = start_time;
    int image_count = 0;

    while (current_time - start_time < time_limit) {
        generate_mandelbrot_image(colors, palette);

        char filename[64];
        snprintf(filename, sizeof(filename), "..images/mandelbrot_CPU1000.jpg");
        stbi_write_jpg(filename, X, Y, 3, colors, 100);

        image_count++;
        current_time = omp_get_wtime();
    }

    double elapsed_time = current_time - start_time;
    printf("Time taken: %f seconds\n", elapsed_time);
    printf("Images generated: %d\n", image_count);

    free(palette);
    free(colors);

    return 0;
}

Color* make_palette(int size) {
    Color (*palette) = malloc(sizeof(Color[size + 1]));
    for (int i = 0; i < size + 1; i++) {
        if (i >= size) {
            palette[i] = (Color){.r = 0, .g = 0, .b = 0};
            continue;
        }
        double j;
        if (i == 0) {
            j = 3.0;
        } else {
            j = 3.0 * (log(i) / log(size - 1.0));
        }

        if (j < 1) {
            palette[i] = (Color){
                .r = 255 * j,
                .g = 0,
                .b = 255 * j
            };
        } else if (j < 2) {
            palette[i] = (Color){
                .r = 255,
                .g = 255 * (j - 1),
                .b = 255,
            };
        } else {
            palette[i] = (Color){
                .r = 255 * (j - 2),
                .g = 255,
                .b = 255 * (j - 2),
            };
        }
    }
    return palette;
}
