#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "main.h"

/*
* Student names: Giovanni Kastanja, Derk Barten
*
* Build command: make
*
* Usage:
* ./eratosthenes [limit] [batch_size]
* limit: how many primes are generated (default: infinite) 
* batch_size: how many numbers the generator generates at once (default: 1)
*/

int batch_size = 1;

// count the number of primes
pthread_mutex_t prime_mutex;
int primes = 0;
int limit = 0;

int main(int argc, char *argv[]){
    // Use the limit given by the user
    if (argc > 1) {
        limit = atoi(argv[1]);
    }
    if (argc > 2) {
        batch_size = atoi(argv[2]);
    }

    // Create a queue to connect the generator with the first filter
    struct queue outbound_queue;
    outbound_queue.first = NULL;
    outbound_queue.last = NULL;
    // Start of with the first prime
    int value = 2;

    // Create a mutex and condition for the first filter
    pthread_mutex_t out_mutex;
    pthread_cond_t out_cond;
    pthread_mutex_init(&out_mutex, NULL);
    pthread_cond_init(&out_cond, NULL);
    // Initiate the mutex on the prime counter
    pthread_mutex_init(&prime_mutex, NULL);

    // Put the arguments of the filter function in a struct
    FilterArgs filter_args;
    filter_args.associated_prime = value;
    filter_args.inbound_queue = &outbound_queue;
    filter_args.in_mutex = &out_mutex;
    filter_args.in_cond = &out_cond;
    create_filter(&filter_args);
    
    // Start the number generator loop
    while (1) {
        // Lock the queue when the generator is pushing a value
        pthread_mutex_lock(&out_mutex);
        int batch= value + batch_size;
        int send_signal = !is_empty(&outbound_queue);
        for (; value < batch; value++) {            
            push(&outbound_queue, value);      
        }
        // Signal the filter it may continue taking values out of the queue
        if (send_signal) {
            pthread_cond_broadcast(&out_cond);
        }
        pthread_mutex_unlock(&out_mutex);
    }
}

// Spawn a filter thread
void create_filter(FilterArgs *filter_args) {
    pthread_t *filter_tid = (pthread_t *)malloc(sizeof(pthread_t));
    pthread_create(filter_tid, NULL, filter, (void *)filter_args);
}

// The filter representation
void *filter(void* pargs) {
    // Cast the void pointer to  the correct struct type
    FilterArgs* filter_args = (FilterArgs*) pargs;
    // State that this is the newest filter
    int has_created = 0;
    struct queue* inbound_queue = filter_args->inbound_queue;
    int prime = filter_args->associated_prime;
    pthread_mutex_t *in_mutex = filter_args->in_mutex; 
    pthread_cond_t *in_cond = filter_args->in_cond;

    // Create a new queue on the stack for the next filter
    struct queue outbound_queue;
    outbound_queue.first = NULL;
    outbound_queue.last = NULL;
    int value;

    // Create the mutexes for the next filter
    pthread_mutex_t out_mutex;
    pthread_cond_t out_cond;
    pthread_mutex_init(&out_mutex, NULL);
    pthread_cond_init(&out_cond, NULL);

    // Start the filter loop
    while (1) {
        // Lock the queue when we pop a value
        pthread_mutex_lock(in_mutex);
        if (!is_empty(inbound_queue)) {
            value = pop(inbound_queue);
            pthread_mutex_unlock(in_mutex);
            if (value == -1) {
                printf("Error encountered in pop\n");
                exit(1);
            }
        }
        else {
            // Wait for the queue to have values again
            pthread_cond_wait(in_cond, in_mutex);
            pthread_mutex_unlock(in_mutex);
            continue;
        }

        // If the number is divisable, discard
        if (value % prime == 0)
            continue;
        else {
            // Check if the filter is at the end of the chain
            if (!has_created) {
                // Create the arguments of the new filter
                FilterArgs new_filter_args;
                new_filter_args.associated_prime = value;
                new_filter_args.inbound_queue = &outbound_queue;
                new_filter_args.in_mutex = &out_mutex;
                new_filter_args.in_cond = &out_cond;

                // Increase the prime counter
                pthread_mutex_lock(&prime_mutex);
                primes++;
                pthread_mutex_unlock(&prime_mutex);

                // Exit if the limit is reached
                // The limit is 0 by default and will therefore never exit
                if (primes > limit && limit != 0) {
                    exit(0);
                }

                // Print out the prime number
                printf("%i ", value);
                fflush(stdout);

                create_filter(&new_filter_args);
                has_created = 1;
            }
            // Write to the outbound queue
            pthread_mutex_lock(&out_mutex);
            if (is_empty(&outbound_queue)) {
                push(&outbound_queue, value);
                // Signal the filter it may continue taking values out of the queue
                pthread_cond_broadcast(&out_cond);
            }
            else {
                push(&outbound_queue, value);
            }
            pthread_mutex_unlock(&out_mutex);
        }
    }
}
 