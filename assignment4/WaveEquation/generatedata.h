#pragma once

typedef float (*func_t)(float x);

float gauss(float x);
void fill(float *array, int offset, int range, float sample_start,
        float sample_end, func_t f);
