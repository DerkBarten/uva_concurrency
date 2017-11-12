/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

#include "simulate.h"

double WAVE_C = 0.15; 

/* Add any functions you may need (like a worker) here. */
double wave(double *old_array, double *current_array, int i, int i_max) {
    // Both endpoints are always zero
    if (i == 0 || i == i_max - 1){
        return 0;
    }
    // Can threads read same memory address at the same time?
    return 2 * current_array[i] - old_array[i] + 
           WAVE_C * (current_array[i - 1] - 
           (2 * current_array[i] - current_array[i + 1]));
}

/*
 * Executes the entire simulation.
 *
 * Implement your code here.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * num_threads: how many threads to use
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int i_max, const int t_max, const int num_threads,
        double *old_array, double *current_array, double *next_array)
{
    /*
     * Your implementation should go here.
     */ 
     int i,j; 
    omp_set_num_threads(num_threads); // set the number of threads omp will create
    #pragma omp parallel for private(i,j) firstprivate(old_array, current_array, next_array) if(t_max >= 20000) // j is private to the inner loop
    // experiment with differnt scheduling
    for (i = 0; i < t_max; i++) {
        for (j = 0; j < i_max; j++){
            next_array[j] = wave(old_array, current_array, j, i_max);   
        }
    }

    return current_array;
}
