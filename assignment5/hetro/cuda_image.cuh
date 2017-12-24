#include "image.h"

void cuda_grayscale(image_t *input, image_t *output);

void cuda_contrast(image_t *image, int mean);

void cuda_smoothing(image_t *image, image_t *original);