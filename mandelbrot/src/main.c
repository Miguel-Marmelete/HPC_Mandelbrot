#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <rgb.h>
#include <mandelbrotCPU.h>
#include <mandelbrotGPU.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <cuda_runtime.h>


#define WIDTH 7680
#define HEIGHT 4230
#define MAX_ITER 500

void mandelbrotCPU();
void mandelbrotGPU();
int main() {
    int flag = 0;
    while(flag == 0){
        printf("Paralelismo com CPU: Insira 1 \n");
        printf("Paralelismo com GPU: Insira 2 \n");
        int option;
        scanf_s("%d", &option);
        if(option != 1 && option != 2){
            printf("Opção Inválida \n");
            continue;
        }
        if(option == 1 ){
            flag = 1;
            mandelbrotCPU();
        }
        if(option == 2){
            flag = 1;
            mandelbrotGPU();
        }
    }
    return 0;
    
}
void mandelbrotCPU(){
   
    RGB *image = (RGB *)malloc(sizeof(RGB) * WIDTH * HEIGHT);
    if (!image) {
        perror("Unable to allocate memory");
        exit(1);
    }

    clock_t start, end;
    double cpu_time_used;
    start = clock();

    #pragma omp parallel for collapse(5)
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            image[y * WIDTH + x] = mandelbrot(x, y, WIDTH, HEIGHT , MAX_ITER);
        }
    }
    end = clock(); // Marca o tempo final


    stbi_write_jpg("../images/mandelbrot.jpg", WIDTH, HEIGHT, sizeof(RGB), image, 100); // Quality set to 100
    free(image);
    
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Tempo de execucao: %f segundos\n", cpu_time_used);
}
void mandelbrotGPU(){
    
    RGB *image = (RGB *)malloc(sizeof(RGB) * WIDTH * HEIGHT);
    RGB *d_image;

    cudaMalloc((void **)&d_image, sizeof(RGB) * WIDTH * HEIGHT);

    clock_t start, end;
    double cpu_time_used;
    start = clock();
    mandelbrot_kernel<<<1, 1>>>(d_image);
    end = clock(); // Marca o tempo final

    cudaMemcpy(image, d_image, sizeof(RGB) * WIDTH * HEIGHT, cudaMemcpyDeviceToHost);
    stbi_write_jpg("../images/mandelbrot.jpg", WIDTH, HEIGHT, sizeof(RGB), image, 100); // Quality set to 100


    cudaFree(d_image);
    free(image);

    
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Tempo de execucao: %f segundos\n", cpu_time_used);
}
    

