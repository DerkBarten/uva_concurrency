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
    int t_max;
    int num_threads;
    int thread_number;
} ThreadArgs;

int counter = 0;
pthread_mutex_t mutex;
pthread_mutex_t mutex_main;
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

    for (int i = 0; i < args->t_max; i++) {
        printf("Thread %i starts cycle %i\n", args->thread_number, i);
        for (int j = args->start; j < args->end; j++){
            next_array[j] = wave(old_array, current_array, j, args->i_max);
        }
        pthread_mutex_lock(&mutex);
        counter++;
        if (counter == args->num_threads) {
            pthread_cond_broadcast(&count_cond);
            printf("I send the signal: %i\n", args->thread_number);
        }
        // wait for every other thread to finish
        printf("Thread %i ended cycle %i\n", args->thread_number, i);
        pthread_cond_wait(&sync_cond, &mutex);
    }
    printf("Thread %i exited\n", args->thread_number);
    // Free the malloc before we exit
    free(args);
    pthread_exit(0);
}

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
    // num_threads - 1 full threads, 1 leftover thread
    int chunk_size = i_max / (num_threads - 1);
    int leftover = i_max % (num_threads - 1);

    pthread_mutex_init(&mutex, NULL);
    pthread_mutex_init(&mutex_main, NULL);
    pthread_cond_init(&sync_cond, NULL);
    pthread_cond_init(&count_cond, NULL);

    printf("chunck size: %i leftover: %i\n", chunk_size, leftover);

    pthread_t tid[num_threads];
    for (int i = 0; i < num_threads - 1; i++) {
        ThreadArgs *args = (ThreadArgs *)malloc(sizeof(ThreadArgs));
        args->old_array = old_array;
        args->current_array = current_array;
        args->next_array = next_array;
        args->start = i * chunk_size;
        args->end = (i + 1) * chunk_size;
        args->i_max = i_max;
        args->t_max = t_max;
        args->num_threads = num_threads;
        args->thread_number = i;

        // We need to cast the argument pointer to a void pointer
        pthread_create(&tid[i], NULL, worker, (void *)args);
        
    }
    // Create a leftover thread here
    ThreadArgs *args = (ThreadArgs *)malloc(sizeof(ThreadArgs));
    args->old_array = old_array;
    args->current_array = current_array;
    args->next_array = next_array;
    args->start = num_threads * chunk_size;
    args->end = (num_threads * chunk_size) + leftover;
    args->i_max = i_max;
    args->t_max = t_max;
    args->num_threads = num_threads;
    args->thread_number = num_threads - 1;
    pthread_create(&tid[num_threads - 1], NULL, worker, (void *)args);

    for (int i = 0; i < t_max; i++) {
        pthread_mutex_lock(&mutex_main);
        printf("Main waits for next cycle signal\n");
        pthread_cond_wait(&count_cond, &mutex_main);
        printf("Main received signal\n");
        counter = 0;
        // Swap arrays around
        printf("Swapping arrays around\n");
        double *temp = old_array;
        old_array = current_array;
        current_array = next_array;
        next_array = temp;
        pthread_cond_broadcast(&sync_cond);
        printf("Main sent continue broadcast\n");
    }
    printf("Main exited loop\n");
    // Wait for all threads to end
    for (int i = 0; i <= num_threads; i++) {
        pthread_join(tid[i], NULL);
    }
    
    /* You should return a pointer to the array with the final results. */
    return current_array;
}
