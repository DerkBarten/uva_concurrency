#include <stdio.h>
#include <math.h>

#include "generatedata.h"

/*
 * Simple gauss with mu=0, sigma^1=1
 */
float gauss(float x)
{
    return exp((-1 * x * x) / 2);
}

/*
 * Fills a given array with samples of a given function. The first sample is
 * placed at array index `offset'. `range' samples are taken, so your array
 * should be able to store at least offset+range doubles. The function `f' is
 * sampled `range' times between `sample_start' and `sample_end'.
 */
void fill(float *array, int offset, int range, float sample_start,
        float sample_end, func_t f)
{
    int i;
    float dx;

    dx = (sample_end - sample_start) / range;
    for (i = 0; i < range; i++) {
        array[i + offset] = f(sample_start + i * dx);
    }
}

