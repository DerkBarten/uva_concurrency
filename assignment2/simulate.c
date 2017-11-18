/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include "simulate.h"

/*
rank 0 is the master, the others send their part of the array to the master
the master writes to disk
*/



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

    MPI_Request reqs[3];
    printf("rank: %i\n", rank);
    printf("numtasks: %i\n", numtasks);

    fflush(stdout);

    int left_neighbor_rank = rank - 1 ;
    int right_neighbor_rank = rank + 1;
    int array_length;
    int array_leftover;

    // If there is just one task
    array_leftover = 0;
    if (numtasks == 1) {
        array_length = i_max;
    }
    // If the tasks are well dividable
    else if (i_max % numtasks == 0) {
        array_length = i_max / numtasks;
        array_leftover = array_length;
    }
    // If the tasks cannot be well divided
    else {
        array_length = i_max / numtasks;
        array_leftover = i_max - (numtasks - 1) * array_length;    
    }
    printf("array length: %i\n", array_length);
    printf("leftover length: %i\n", array_leftover);

    // Calculate where the arrays should point
    my_old_array = old_array + rank * array_length;
    my_current_array = current_array + rank * array_length;
    my_next_array = next_array + rank * array_length;

    float my_left = my_current_array[0];
    float my_right = my_current_array[array_length - 1];

    float left_neighbor_float;
    float right_neighbor_float;

    // When there is just one worker
    if (numtasks == 1) {
        for (int t = 0; t < t_max; t++) {
            for (int i = 0; i < array_length; i++) {
                my_next_array[i] = wave(i);
            }
            buffer_swap();
        }
        MPI_Finalize();
        return my_current_array;
    }
    // The right most chunk
    else if (rank == numtasks - 1) {
        printf("Im the right most chunk: %i\n", rank);
        for (int t = 0; t < t_max; t++) {
            MPI_Isend(&my_left, 1, MPI_FLOAT, left_neighbor_rank, 1, MPI_COMM_WORLD, &reqs[0]);
            
            for (int i = 1; i < array_leftover - 1; i++) {
                my_next_array[i] = wave(i);
            }

            MPI_Recv(&left_neighbor_float, 1, MPI_FLOAT, left_neighbor_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Set the right value
            my_current_array[array_leftover - 1] = 0.0;
            my_next_array[array_leftover - 1] = 0.0;

            // Set the neighbouring current value
            my_current_array[-1] = left_neighbor_float;
            // Calculate the left most cell
            my_next_array[0] = wave(0);
            
            buffer_swap();
        }   
        MPI_Isend(my_current_array, array_leftover, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &reqs[1]); 
        printf("%i: sent my chunk\n", rank);
        fflush(stdout); 
    }
    // The left most chunk
    else if (rank == 0) {
        for (int t = 0; t < t_max; t++) {
            MPI_Isend(&my_right, 1, MPI_FLOAT, right_neighbor_rank, 2, MPI_COMM_WORLD, &reqs[0]);
            
            for (int i = 1; i < array_length - 1; i++) {
                my_next_array[i] = wave(i);
            }

            MPI_Recv(&right_neighbor_float, 1, MPI_FLOAT, right_neighbor_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Set the left values
            my_current_array[0] = 0.0;
            my_next_array[0] = 0.0;

            // set the neighbouring current value
            my_current_array[array_length] = right_neighbor_float;
            my_next_array[array_length - 1] = wave(array_length - 1);

            buffer_swap();
        }
        printf("Started gathering chunks\n");
        fflush(stdout);

        for (int i = 1; i < numtasks - 1; i++) {
            MPI_Recv((double*)(current_array + i * array_length), array_length, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Received chunk %i\n", i);
        }
        MPI_Recv((double*)(current_array + (numtasks - 1) * array_length), array_leftover, MPI_DOUBLE, numtasks - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Received last chunk %i\n", numtasks - 1);
        MPI_Finalize();
        return current_array;
    }
     // The chunks that are not at the edges
    else {
        printf("Im a middle chunk: %i\n", rank);
        for (int t = 0; t < t_max; t++) {
            MPI_Isend(&my_left, 1, MPI_FLOAT, left_neighbor_rank, 1, MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(&my_right, 1, MPI_FLOAT, right_neighbor_rank, 2, MPI_COMM_WORLD, &reqs[1]);

            for (int i = 1; i < array_length - 1; i++) {
                my_next_array[i] = wave(i);
            }

            MPI_Recv(&left_neighbor_float, 1, MPI_FLOAT, left_neighbor_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&right_neighbor_float, 1, MPI_FLOAT, right_neighbor_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Left side
            my_current_array[-1] = left_neighbor_float;
            my_next_array[0] = wave(0);

            // Right side
            my_current_array[array_length] = right_neighbor_float;
            my_next_array[array_length - 1] = wave(array_length - 1);

            buffer_swap();
        }
        MPI_Isend(my_current_array, array_length, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &reqs[2]);
        printf("%i: sent my chunk\n", rank);
        fflush(stdout); 
    }

    MPI_Finalize();
    return NULL;
}
