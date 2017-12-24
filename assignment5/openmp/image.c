#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "image.h"
#include "timer.h"

#include <math.h>
#include "omp.h"

/* Load the image specified by the filename*/
int load_image(char *filename, image_t *image) {
    image->data = stbi_load(filename, &image->w, &image->h, &image->n, 0);
    if (image->data != NULL) {
        return 1;
    }
    return 0;    
}

/* Write the image object to the specified filename */
int save_image(char *filename, image_t *image) {
    stbi_write_png(filename, image->w, image->h, image->n, image->data, image->w * image->n);
}

/* Free the data array in the image object */ 
void unload_image(image_t *image) {
    stbi_image_free(image->data);
}

/* Convert a rgb image to a grayscale image */
int openmp_grayscale(image_t *input, image_t *output) {
    if (input->n != 3 && input->n != 4) {

        printf("number of channels: %d", input->n);
        printf("ERROR: This function only supports 3 and 4 channels\n");
        return 0;
    }

    double time;
    // Initialize the output image object
    output->data = (byte*)malloc(sizeof(byte) * input->w * input->h);
    output->w = input->w;
    output->h = input->h;
    output->n = 1;


    int i, j;
    byte r,g,b,gray;
    #pragma omp parallel for collapse(2) private(r,g,b,gray,j) shared(output)
    for (i = 0; i < input->h; i++) {
        for (j = 0; j < input->w; j++) {
             r = input->data[(i * input->w + j) * input->n];
             g = input->data[(i * input->w + j) * input->n + 1];
             b = input->data[(i * input->w + j) * input->n + 2];

            // Calculate the grayscale value
            gray = (float)r * 0.299f + (float)g * 0.587f + (float)b * 0.114f;
            // Set the corresponding output pixel to the grayscale value
            output->data[i * input->w + j] = gray;
        }
    }
    return 1;
}

int openmp_contrast(image_t *image) {
    if (image->n != 1) {
        return 0;
    }
    double time;
    int brightness = 0;
    int size = image->w * image->h;
    int i;

    for (i = 0; i < size; i++) {
        brightness += image->data[i];
    }
    float mean = floor((float)brightness / (float)size) / 255.0;
    float value;

    #pragma omp parallel for private(value) firstprivate(image)
    for (i = 0; i < size; i++) {
        value = image->data[i] / 255.0; 
        if (value > mean) {
            image->data[i] = (byte)((pow(value - mean, 0.5) / pow(1.0 - mean, 0.5)) * 255.0);
        }
        else {
            image->data[i] = 0;
        }
    }
}

int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

int openmp_smoothing(image_t *image) {
    byte T[5][5] = {{1, 2, 3, 2, 1},
                             {2, 4, 6, 4, 2},
                             {3, 6, 9, 6, 3},
                             {2, 4, 6, 4, 2},
                             {1, 2, 3, 2, 1}};
    int i,j,k,l;
    double time;
    unsigned int sum;
    
    #pragma omp parallel for collapse(2) private(j,k,l, sum) firstprivate(image)
    for ( i = 0; i < image->h; i++) {
        for ( j = 0; j < image->w; j++) {    
            sum = 0;
            // Loop over the neighbourhood
            for ( k = 0; k < 5; k++) {
                for (l = 0; l < 5; l++) {
                    // reduction 
                    sum += T[k][l] * 
                    image->data[mod(i + k - 2, image->h) * image->w  + mod(j + l - 2, image->w)];
                }
            }
            image->data[i * image->w + j] = sum / 81;
        }
    }
}