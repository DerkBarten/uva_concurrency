#include "image.h"

int openmp_grayscale(image_t *input, image_t *output);

int openmp_contrast(image_t *input, int mean);

int openmp_smoothing(image_t *input, image_t *original);