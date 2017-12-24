#ifndef IMAGE_H
#define IMAGE_H

typedef unsigned char byte;

typedef struct {
    // Width, height, channels
    int w, h, n;
    // Pointer to the image data
    byte *data;
} image_t;


int load_image(char *filename, image_t *image);

int save_image(char *filename, image_t *image);

void unload_image(image_t *image);

void split_image(float ratio, image_t *input, image_t *part1, image_t *part2);
#endif