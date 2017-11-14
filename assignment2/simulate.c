/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "simulate.h"



/* Add any global variables you may need. */


/* Add any functions you may need (like a worker) here. */
double wave(int i, int i_max) {
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
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int i_max, const int t_max, double *old_array,
        double *current_array, double *next_array)
{
    double left_current;
    double right_current;

    int numtasks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Request reqs[4];

    int left_neighbor = rank - 1 ;
    int right_neighbor = rank + 1;

    int array_normal_length = i_max / (numtasks - 1);
    int array_leftover_length = i_max % (numtasks - 1); 

    double *my_old_array = old_array + rank * array_normal_length;
    double *my_current_array = current_array + rank * array_normal_length;
    double *my_next_array = next_array + rank * array_normal_length;

    if (left_neighbor > 0 && right_neighbor < numtasks) {
        // somewhere in the middle
        // send to process left
        MPI_FLOAT left = my_current_array[0];
        MPI_FLOAT right = my_current_array[array_normal_length - 1]
        // TODO: check tags
        MPI_ISend(&left, 1, MPI_FLOAT, left_neighbor, 1, MPI_COMM_WORLD, &reqs[0]);
        MPI_ISend(&right, 1, MPI_FLOAT, right_neighbor, 2, MPI_COMM_WORLD, &reqs[1]);

        // wait for left and right
        MPI_Recv();
        MPI_Recv();

    }
    /*
     * Your implementation should go here.
     */

    /* You should return a pointer to the array with the final results. */
    return current_array;
}
