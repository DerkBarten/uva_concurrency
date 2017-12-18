#ifndef IMAGE_H
#define IMAGE_H

typedef unsigned char byte;

typedef struct {
    // Width, height, channels
    int w, h, n;
    // Pointer to the image data
    unsigned char *data;
} image_t;


int load_image(char *filename, image_t *image);

int save_image(char *filename, image_t *image);

void unload_image(image_t *image);

//int rgb_to_grayscale(image_t *input, image_t *output);

int contrast_modification(image_t *input);

int triangular_smoothing(image_t *image);
#endif