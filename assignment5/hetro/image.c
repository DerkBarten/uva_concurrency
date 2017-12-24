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
    free(image->data);
}

/* Split image horizontally in two parts */
void split_image(float ratio, image_t *input, image_t *part1, image_t *part2) {
    int split = input->h * ratio;

    // Top part of image
    part1->w = input->w;
    part1->h = split;
    part1->n = input->n;
    part1->data = input->data;

    // Bottom part of image
    part2->w = input->w;
    part2->h = input->h - split;
    part2->n = input->n;
    part2->data = input->data + (input->w * split * input->n);
}