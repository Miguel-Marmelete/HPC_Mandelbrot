#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h> // Biblioteca para OpenMP
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

// Estrutura para representar cor RGB
typedef struct {
    uint8_t r;
    uint8_t g;
    uint8_t b;
} RGB;

// Função para calcular a cor de um ponto no conjunto de Mandelbrot
RGB mandelbrot(int x, int y, int WIDTH, int HEIGHT, int MAX_ITER) {
    // Mapear coordenadas de pixel para o plano complexo
    double creal = (x - WIDTH / 2.0) * 4.0 / WIDTH;
    double cimag = (y - HEIGHT / 2.0) * 4.0 / HEIGHT;
    double real = 0, imag = 0;
    int n = 0;

    // Loop de cálculo do Mandelbrot
    while (real * real + imag * imag <= 4 && n < MAX_ITER) {
        double temp = real * real - imag * imag + creal;
        imag = 2 * real * imag + cimag;
        real = temp;
        n++;
    }

    // Atribuir cor com base na contagem de iterações
    RGB color;
    if (n == MAX_ITER) {
        // Preto para pontos dentro do conjunto
        color.r = 0;
        color.g = 0;
        color.b = 0;
    } else {
        // Cor com base na contagem de iterações
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

    // Dimensões da imagem e iterações máximas
    int WIDTH = 7680;
    int HEIGHT = 4320;
    int MAX_ITER = 100;
    double max_time = 60.0; // Tempo máximo de execução
    int images_generated = 0;

    start = omp_get_wtime(); // Marcar o início do tempo de execução
    while (cpu_time_used < max_time) {
	
        // Alocar memória para a imagem completa
        RGB *full_image = (RGB *)malloc(sizeof(RGB) * WIDTH * HEIGHT);
        if (!full_image) {
            perror("Unable to allocate memory");
            exit(1);
        }

        // Calcular a imagem Mandelbrot em paralelo usando OpenMP e GPUs
        int num_devices = 4500; // Número de GPUs disponíveis
        int rows_per_device = HEIGHT / num_devices;

        #pragma omp parallel num_threads(num_devices)
        {
            int device_id = omp_get_thread_num();
            int start_row = device_id * rows_per_device;
            int end_row = (device_id == num_devices - 1) ? HEIGHT : start_row + rows_per_device;

            RGB *partial_image = (RGB *)malloc(sizeof(RGB) * WIDTH * (end_row - start_row));
            if (!partial_image) {
                perror("Unable to allocate memory");
                exit(1);
            }

            #pragma omp target teams distribute parallel for collapse(2) map(to: WIDTH, HEIGHT, MAX_ITER) map(from: partial_image[0:WIDTH * (end_row - start_row)]) device(device_id)
            for (int y = start_row; y < end_row; y++) {
                for (int x = 0; x < WIDTH; x++) {
                    partial_image[(y - start_row) * WIDTH + x] = mandelbrot(x, y, WIDTH, HEIGHT, MAX_ITER);
                }
            }

            // Copiar a parte calculada para a imagem completa
            #pragma omp critical
            {
                memcpy(&full_image[start_row * WIDTH], partial_image, sizeof(RGB) * WIDTH * (end_row - start_row));
            }
            free(partial_image);
        }

        // Salvar a imagem em um arquivo JPEG
        char filename[256];
        snprintf(filename, sizeof(filename), "../images/mandelbrotGPU.jpg");
        stbi_write_jpg(filename, WIDTH, HEIGHT, sizeof(RGB), full_image, 100);
        images_generated++;

        end = omp_get_wtime(); // Marcar o fim do tempo de execução
        cpu_time_used = end - start; // Calcular o tempo de execução
        free(full_image); // Liberar a memória da imagem completa
    }

    // Imprimir o tempo total de execução e o número de imagens geradas
    printf("Execution time: %f seconds\n", cpu_time_used);
    printf("Images generated: %d\n", images_generated);

    return 0;
}
