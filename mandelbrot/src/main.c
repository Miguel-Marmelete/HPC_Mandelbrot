#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main() {
    // dimensões da imagem
    int width = 640;
    int height = 480;
    int channels = 3; // RGB

   
    unsigned char *image = malloc(width * height * channels);
    if (image == NULL) {
        printf("Erro ao alocar memória para a nova imagem.\n");
        return 1;
    }

    
    memset(image, 0, width * height * channels);



    //guardar imagem
    if (!stbi_write_jpg("../images/mandelbrot.jpg", width, height, channels, image, 100)) {
        printf("Erro ao salvar a nova imagem.\n");
        free(image);
        return 1;
    }

    free(image);
    return 0;
}
