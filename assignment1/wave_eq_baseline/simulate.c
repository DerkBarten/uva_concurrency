/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>

#include "simulate.h"

double WAVE_C = 0.15; 

/* Add any functions you may need (like a worker) here. */
double wave(double *old_array, double *current_array, int i, int i_max) {
    // Both endpoints are always zero
    if (i == 0 || i == i_max - 1){
        return 0.0;
    }

    return 2.0 * current_array[i] - old_array[i] + 
           WAVE_C * (current_array[i - 1] - 
           (2.0 * current_array[i] - current_array[i + 1]));
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
    /*
    for (int j = 0; j < i_max; j++){
        printf("o%i %f\n",j,old_array[j]);
    }
    for (int j = 0; j < i_max; j++){
        printf("c%i %f\n",j,current_array[j]);
    }
    for (int j = 0; j < i_max; j++){
        next_array[j] = wave(old_array, current_array, j, i_max);
    }
    for (int j = 0; j < i_max; j++){
        printf("n%i %f\n",j,next_array[j]);
    }
    */

    for (int i = 0; i < t_max; i++) {
        for (int j = 0; j < i_max; j++){
            next_array[j] = wave(old_array, current_array, j, i_max);
        }
        /*
        printf("before:\n");
        printf("old: %p\n", old_array);
        printf("cur: %p\n", current_array);
        printf("nex: %p\n", next_array);
        */
        double *temp = old_array;
        old_array = current_array;
        current_array = next_array;
        next_array = temp;

        /*
        printf("after:\n");
        printf("old: %p\n", old_array);
        printf("cur: %p\n", current_array);
        printf("nex: %p\n", next_array);
        */
    }

    return current_array;
}
