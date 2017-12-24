#include "image.h"
#include "openmp_image.h"
#include "cuda_image.cuh"
#include <stdlib.h>
#include <stdio.h>

int image_mean(image_t *image) {
    int brightness = 0;
    int size = image->w * image->h;
    int i;

    for (i = 0; i < size; i++) {
        brightness += image->data[i];
    }

    return brightness / size;
}

void hetro_image(image_t *input, image_t *output) {
    image_t cuda_input, openmp_input;
    image_t cuda_output, openmp_output;

    output->data = (byte*)malloc(sizeof(byte) * input->w * input->h);
    output->w = input->w;
    output->h = input->h;
    output->n = 1;

    split_image(0.5, input, &cuda_input, &openmp_input);
    split_image(0.5, output, &cuda_output, &openmp_output);

    cuda_grayscale(&cuda_input, &cuda_output);
    openmp_grayscale(&openmp_input, &openmp_output);

    int mean = image_mean(output);

    cuda_contrast(&cuda_output, mean);
    openmp_contrast(&openmp_output, mean);

    cuda_smoothing(&cuda_output, output);
    openmp_smoothing(&openmp_output, output);
}