#include <stdio.h>
#include <stdlib.h>
#include "simulate.h"
#include "vector-math.h"


float *simulate(const int i_max, const int t_max, const int num_threads,
                float *old_array, float *current_array, float *next_array) {
    vectorSimulateCuda(i_max, old_array, current_array, next_array, t_max); 
    return current_array;
}