#include "image.h"
#include "openmp_image.h"
#include <math.h>
#include <stdio.h>
#include "omp.h"

/* Convert a rgb image to a grayscale image */
int openmp_grayscale(image_t *input, image_t *output) {
    if (input->n != 3 && input->n != 4) {

        printf("number of channels: %d", input->n);
        printf("ERROR: This function only supports 3 and 4 channels\n");
        return 0;
    }

    double time;

    int i, j;
    byte r,g,b,gray;
    #pragma omp parallel for private(r,g,b,gray,j) shared(output)
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

int openmp_contrast(image_t *image, int mean) {
    if (image->n != 1) {
        return 0;
    }

    float value;
    int size = image->w * image->h;

    #pragma omp parallel for private(value) firstprivate(image)
    for (int i = 0; i < size; i++) {
        value = image->data[i]; 
        if (value > mean) {
            float d1 = (float)(value - mean) / 255.0f;
            float d2 = 1.0f - ((float)mean / 255.0f);
            image->data[i] = (byte)((pow(d1, 0.5f) / pow(d2, 0.5f))* 255.0f);
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

int openmp_smoothing(image_t *image, image_t *original) {
    byte T[5][5] = {{1, 2, 3, 2, 1},
                             {2, 4, 6, 4, 2},
                             {3, 6, 9, 6, 3},
                             {2, 4, 6, 4, 2},
                             {1, 2, 3, 2, 1}};
    int i,j,k,l;
    double time;
    unsigned int sum;
    #pragma omp parallel for private(j,k,l) firstprivate(image) reduction(+:sum)
    for ( i = 0; i < image->h; i++) {
        for ( j = 0; j < image->w; j++) {    
            sum = 0;
            // Loop over the neighbourhood
            for ( k = 0; k < 5; k++) {
                for (l = 0; l < 5; l++) {
                    // reduction 
                    sum += T[k][l] * 
                    image->data[((i + k - 2) % original->h) * original->w  + mod(j + l - 2, original->w)];
                }
            }

            image->data[i * original->w + j] = sum / 81;
        }
    }
}