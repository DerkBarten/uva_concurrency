#include "image.h"
#include "openmp_image.h"
#include "cuda_image.cuh"
#include <stdlib.h>

void hetro_image(image_t *input, image_t *output) {
    image_t part1_in, part2_in;
    image_t part1_out, part2_out;

    output->data = (byte*)malloc(sizeof(byte) * input->w * input->h);
    output->w = input->w;
    output->h = input->h;
    output->n = 1;

    split_image(0.5, input, &part1_in, &part2_in);
    split_image(0.5, output, &part1_out, &part2_out);

    cuda_image(&part1_in, &part1_out, input);
    openmp_image(&part2_in, &part2_out, input);
}