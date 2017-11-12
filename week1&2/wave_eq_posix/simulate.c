/*
 * simulate.c
 *
 * Student names: Giovanni Kastanja, Derk Barten
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "simulate.h"

// Please uncomment this to show debugging statements
//# define DEBUG

#ifdef DEBUG
# define DEBUG_PRINT(x) printf x
#else
# define DEBUG_PRINT(x)
#endif

/* Add any global variables you may need. */
double WAVE_C = 0.15; 

typedef struct ThreadArgs{
    int start;
    int end;
    int i_max;
    int t_max;
    int num_threads;
    int thread_number;
} ThreadArgs;

int counter = 0;

double *old_array;
double *current_array;
double *next_array;

pthread_mutex_t mutex;
pthread_cond_t sync_cond;
pthread_cond_t count_cond;

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

void* worker(void* pargs) {
    // Cast the void pointer back to an argument pointer
    ThreadArgs* args = (ThreadArgs*) pargs; 

    for (int i = 0; i < args->t_max; i++) {
        DEBUG_PRINT(("Thread %i starts cycle %i\n", args->thread_number, i));

        for (int j = args->start; j < args->end; j++){
            next_array[j] = wave(j, args->i_max);
        }
        pthread_mutex_lock(&mutex);
        counter++;
        if (counter == args->num_threads) {
            pthread_cond_broadcast(&count_cond);
            DEBUG_PRINT(("I send the signal: %i\n", args->thread_number));
        }

        // wait for every other thread to finish
        DEBUG_PRINT(("Thread %i ended cycle %i, cnt: %i\n", args->thread_number, i, counter));
        pthread_cond_wait(&sync_cond, &mutex);
        pthread_mutex_unlock(&mutex);
    }
    DEBUG_PRINT(("Thread %i exited\n", args->thread_number));
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
        double *old_array_, double *current_array_, double *next_array_)
{
    old_array = old_array_;
    current_array = current_array_;
    next_array = next_array_;

    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&sync_cond, NULL);
    pthread_cond_init(&count_cond, NULL);

    int chunk_size;
    int leftover;
    if (num_threads > 1) {
        // num_threads - 1 full threads, 1 leftover thread
        chunk_size = i_max / (num_threads - 1);
        leftover = i_max % (num_threads - 1);
    }
    else {
        chunk_size = 0;
        leftover = i_max;
    }

    DEBUG_PRINT(("chunck size: %i leftover: %i\n", chunk_size, leftover));

    // Lock because we don't want the threads to finish before main is ready to listen
    pthread_mutex_lock(&mutex);

    pthread_t tid[num_threads];
    for (int i = 0; i < num_threads - 1; i++) {
        ThreadArgs *args = (ThreadArgs *)malloc(sizeof(ThreadArgs));
        args->start = i * chunk_size;
        args->end = (i + 1) * chunk_size;
        args->i_max = i_max;
        args->t_max = t_max;
        args->num_threads = num_threads;
        args->thread_number = i;

        DEBUG_PRINT(("Thread %i: start=%i end=%i\n", i, args->start, args->end));
        // We need to cast the argument pointer to a void pointer
        pthread_create(&tid[i], NULL, worker, (void *)args);
        
    }
    // Create a leftover thread here
    ThreadArgs *args = (ThreadArgs *)malloc(sizeof(ThreadArgs));
    args->start = (num_threads - 1) * chunk_size;
    args->end = (num_threads - 1) * chunk_size + leftover;
    args->i_max = i_max;
    args->t_max = t_max;
    args->num_threads = num_threads;
    args->thread_number = num_threads - 1;

    DEBUG_PRINT(("Thread leftover: start=%i end=%i\n", args->start, args->end));
    pthread_create(&tid[num_threads - 1], NULL, worker, (void *)args);

    for (int i = 0; i < t_max; i++) {
        DEBUG_PRINT(("Main waits for next cycle signal\n"));
        pthread_cond_wait(&count_cond, &mutex);
        DEBUG_PRINT(("Main received signal\n"));
        counter = 0;
        DEBUG_PRINT(("Swapping arrays around\n"));
        double *temp = old_array;
        old_array = current_array;
        current_array = next_array;
        next_array = temp;
        pthread_cond_broadcast(&sync_cond);
        DEBUG_PRINT(("Main sent continue broadcast\n"));

    }
    DEBUG_PRINT(("Main exited loop\n"));
    pthread_mutex_unlock(&mutex);
    // Wait for all threads to end
    for (int i = 0; i < num_threads; i++) {
        pthread_join(tid[i], NULL);
    }
    
    DEBUG_PRINT(("All is done\n"));
    /* You should return a pointer to the array with the final results. */
    return current_array;
}
