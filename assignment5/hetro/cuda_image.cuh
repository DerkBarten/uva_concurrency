#include "image.h"

void cuda_grayscale(image_t *input, image_t *output);

void cuda_contrast(image_t *image);

void cuda_smoothing(image_t *image, image_t *original);

void cuda_image(image_t *input, image_t *output, image_t *original);