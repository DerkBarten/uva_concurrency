#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "image.h"

#include <math.h>

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
int rgb_to_grayscale(image_t *input, image_t *output) {
    if (input->n != 3 && input->n != 4) {
        printf("ERROR: This function only supports 3 and 4 channels\n");
    }
    // Initialize the output image object
    output->data = (unsigned char*)malloc(sizeof(unsigned char) * input->w * input->h);
    output->w = input->w;
    output->h = input->h;
    output->n = 1;

    // Iterate over every pixel in the input image
    for (int i = 0; i < input->h; i++) {
        for (int j = 0; j < input->w; j++) {
            unsigned char r = input->data[(i * input->w + j) * input->n];
            unsigned char g = input->data[(i * input->w + j) * input->n + 1];
            unsigned char b = input->data[(i * input->w + j) * input->n + 2];

            // Calculate the grayscale value
            unsigned char gray = (r + g + b) / 3;
            // Set the corresponding output pixel to the grayscale vaue
            output->data[i * input->w + j] = gray;
        }
    }
}