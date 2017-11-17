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
double *my_old_array;
double *my_current_array;
double *my_next_array;


// Assume the caller avoids segmentations errors
double wave(int i) {
    // Can threads read same memory address at the same time?
    return 2 * my_current_array[i] - my_old_array[i] + 
           0.15 * (my_current_array[i - 1] - 
           (2 * my_current_array[i] - my_current_array[i + 1]));
}

void buffer_swap(void) {
    double *temp = my_old_array;
    my_old_array = my_current_array;
    my_current_array = my_next_array;
    my_next_array = temp;
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
    MPI_Init(NULL, NULL);
    int numtasks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Request reqs[2];

    int left_neighbor_rank = rank - 1 ;
    int right_neighbor_rank = rank + 1;

    int array_normal_length;
    if (numtasks > 1) {
        array_normal_length = i_max / (numtasks - 1);
    }
    int array_leftover_length = i_max % (numtasks - 1); 

    my_old_array = old_array + rank * array_normal_length;
    my_current_array = current_array + rank * array_normal_length;
    my_next_array = next_array + rank * array_normal_length;

    float my_left = my_current_array[0];
    float my_right = my_current_array[array_normal_length - 1];

    float left_neighbor_float;
    float right_neighbor_float;

    // The parts somewhere in the middle
    if (left_neighbor_rank >= 0 && right_neighbor_rank < numtasks) {
        for (int t = 0; t < t_max; t++) {
            MPI_Isend(&my_left, 1, MPI_FLOAT, left_neighbor_rank, 1, MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(&my_right, 1, MPI_FLOAT, right_neighbor_rank, 2, MPI_COMM_WORLD, &reqs[1]);

            for (int i = 1; i < array_normal_length - 2; i++) {
                next_array[i] = wave(i);
            }

            // The tags are switched around since left/right is relative
            MPI_Recv(&left_neighbor_float, 1, MPI_FLOAT, left_neighbor_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&right_neighbor_float, 1, MPI_FLOAT, right_neighbor_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            my_current_array[0] = left_neighbor_float;
            my_current_array[array_normal_length - 1] = right_neighbor_float;

            next_array[0] = wave(0);
            next_array[array_normal_length - 1] = wave(array_normal_length - 1);

            buffer_swap();
        }
        
    }
    // The left most part
    else if (left_neighbor_rank < 0) {
        for (int t = 0; t < t_max; t++) {
            MPI_Isend(&my_right, 1, MPI_FLOAT, right_neighbor_rank, 2, MPI_COMM_WORLD, &reqs[0]);
            
            for (int i = 1; i < array_normal_length - 2; i++) {
                next_array[i] = wave(i);
            }

            MPI_Recv(&right_neighbor_float, 1, MPI_FLOAT, right_neighbor_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            my_current_array[0] = 0.0;
            my_current_array[array_normal_length - 1] = right_neighbor_float;
            
            next_array[0] = 0.0;
            next_array[array_normal_length - 1] = wave(array_normal_length - 1);
        }

    }
    // The right most part
    else if (right_neighbor_rank >= numtasks) {
        for (int t = 0; t < t_max; t++) {

            MPI_Isend(&my_left, 1, MPI_FLOAT, left_neighbor_rank, 1, MPI_COMM_WORLD, &reqs[0]);
            
            for (int i = 1; i < array_leftover_length - 2; i++) {
                next_array[i] = wave(i);
            }

            MPI_Recv(&left_neighbor_float, 1, MPI_FLOAT, left_neighbor_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            my_current_array[array_leftover_length - 1] = 0.0;
            my_current_array[0] = left_neighbor_float;

            next_array[0] = wave(0);
            next_array[array_leftover_length - 1] = 0.0;
        }         
    }

    /* You should return a pointer to the array with the final results. */
    return current_array;
}
