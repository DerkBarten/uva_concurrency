#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "image.h"

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
int openMP_grayscale(image_t *input, image_t *output) {
    if (input->n != 3 && input->n != 4) {

        printf("number of channels: %d", input->n);
        printf("ERROR: This function only supports 3 and 4 channels\n");
        return 0;
    }
    // Initialize the output image object
    output->data = (byte*)malloc(sizeof(byte) * input->w * input->h);
    output->w = input->w;
    output->h = input->h;
    output->n = 1;

    // Iterate over every pixel in the input image
    // set the number of threads that we'll use
    //  
    #pragma omp parallel for collapse(2);
    for (int i = 0; i < input->h; i++) {
        for (int j = 0; j < input->w; j++) {
            byte r = input->data[(i * input->w + j) * input->n];
            byte g = input->data[(i * input->w + j) * input->n + 1];
            byte b = input->data[(i * input->w + j) * input->n + 2];

            // Calculate the grayscale value
            byte gray = (r + g + b) / 3;
            // Set the corresponding output pixel to the grayscale vaue
            output->data[i * input->w + j] = gray;
        }
    }
    return 1;
}

int openMP_contrast(image_t *image) {
    if (image->n != 1) {
        return 0;
    }
    int brightness = 0;
    int size = image->w * image->h;
    
    for (int i = 0; i < size; i++) {
        brightness += image->data[i];
    }
    float mean = floor((float)brightness / (float)size) / 255.0;
    float value;

    // TODO: might speedup if no conversions inside loop
    #pragma omp parallel for;
    for (int i = 0; i < size; i++) {
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

int openMP_smoothing(image_t *image) {
    byte T[5][5] = {{1, 2, 3, 2, 1},
                             {2, 4, 6, 4, 2},
                             {3, 6, 9, 6, 3},
                             {2, 4, 6, 4, 2},
                             {1, 2, 3, 2, 1}};

    // Loop over every pixel
    #pragma omp parallel for collapse(4);
    for (int i = 0; i < image->h; i++) {
        #pragma private(j, k, l);
        for (int j = 0; j < image->w; j++) {    
            unsigned int sum = 0;
            // Loop over the neighbourhood
            for (int k = 0; k < 5; k++) {
                for (int l = 0; l < 5; l++) {
                    sum += T[k][l] * 
                    image->data[mod(i + k - 2, image->h) * image->w  + mod(j + l - 2, image->w)];
                }
            }
            image->data[i * image->w + j] = sum / 81;
        }
    }
}