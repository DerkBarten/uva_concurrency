/*
 * simulate.c
 *
 * Implement your (parallel) simulation here!
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "simulate.h"


/* Add any global variables you may need. */
double WAVE_C = 0.15; 

typedef struct ThreadArgs{
    double *old_array;
    double *current_array;
    double *next_array;
    int start;
    int end;
    int i_max;
} ThreadArgs;

int counter = 0;
pthread_mutex_t mutex;
pthread_cond_t sync_cond;
pthread_cond_t count_cond;

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

void* worker(void* pargs) {
    // Cast the void pointer back to an argument pointer
    ThreadArgs* args = (ThreadArgs*) pargs; 

    double* old_array = args->old_array;
    double* current_array = args->current_array;
    double* next_array = args->next_array;

    for (int i = 0; i < t_max; i++) {
        for (int j = args->start; j < args->end; j++){
            next_array[j] = wave(old_array, current_array, j, args->i_max);
        }
        pthread_mutex_lock(&mutex);
        counter++;
        if (counter == num_threads) {
            pthread_cond_signal(&count_cond);
        }
        // wait for every other thread to finish
        pthread_cond_wait(&sync_cond, &mutex);
    }
    // Free the malloc before we exit
    free(args);
    pthread_exit(0);
}

/*
// TODO: make this function swap the arrays
// control thread?
void* count_checker() {
    for (int i = 0; i < t_max; i++) {
        pthread_mutex_lock(&mutex);
        pthread_cond_wait(&count_cond, &mutex);
        pthread_mutex_lock(&mutex);
        counter = 0;

        pthread_mutex_unlock(&mutex);
        pthread_cond_signal(&sync_cond);
    }
    pthread_exit(0);
}
*/

/*
 * Executes the entire simulation.
 *
 * Implement your code here.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * num_threads: how many threads to use (excluding the main threads)
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 */
double *simulate(const int i_max, const int t_max, const int num_threads,
        double *old_array, double *current_array, double *next_array)
{
    // Split up the i_max in ~equal parts by the number of threads
    // add a worker which cleans up the small remainder part of the devision is not mod 0
    // manually add a zero to the end of next array since this is not covered by the workers

    
    int chunk_size = i_max / num_threads;
    int leftover = i_max % num_threads;

    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&syn_cond, NULL);
    pthread_cond_init(&count_cond, NULL);

    printf("chunck size: %i leftover: %i\n", chunk_size, leftover);

    pthread_t tid[num_threads + 1];
    for (int i = 0; i < num_threads; i++) {
        ThreadArgs *args = (ThreadArgs *)malloc(sizeof(ThreadArgs));
        args->old_array = old_array;
        args->current_array = current_array;
        args->next_array = next_array;
        args->start = i * chunk_size;
        args->end = (i + 1) * chunk_size;
        args->i_max = i_max;

        // We need to cast the argument pointer to a void pointer
        pthread_create(&tid[i], NULL, worker, (void *)args);
    }
    // TODO: create a leftover thread here
    // pthread_create(&tid[num_threads], NULL, count_checker);

    for (int i = 0; i < t_max; i++) {
        pthread_mutex_lock(&mutex);
        pthread_cond_wait(&count_cond, &mutex);
        pthread_mutex_lock(&mutex);
        counter = 0;
        // Swap arrays around
        pthread_mutex_unlock(&mutex);
        pthread_cond_signal(&sync_cond);
    }

    // Wait for all threads to end
    for (int i = 0; i < num_threads; i++) {
        pthread_join(tid[i], NULL);
    }
    
    /* You should return a pointer to the array with the final results. */
    return current_array;
}
