#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "main.h"

# define BATCH 100

/*
TODO's:

- Check if the generate number function works
- Every queue should have a mutex and cond value
- This whole beast should be made threadsafe

*/



int main(int argc, char *argv[]){
    struct queue outbound_queue;
    outbound_queue.first = NULL;
    outbound_queue.last = NULL;
    int value = 2;

    pthread_mutex_t out_mutex;
    pthread_cond_t out_cond;
    pthread_mutex_init(&out_mutex, NULL);
    pthread_cond_init(&out_cond, NULL);

    FilterArgs filter_args;
    filter_args.associated_prime = value;
    filter_args.inbound_queue = &outbound_queue;
    filter_args.in_mutex = &out_mutex;
    filter_args.in_cond = &out_cond;
    create_filter(&filter_args);
    

    while (1) {
        pthread_mutex_lock(&out_mutex);
        int send_signal = !is_empty(&outbound_queue);
        int limit = value + BATCH;
        
        for (; value < limit; value++) {            
            push(&outbound_queue, value);      
        }
        // Send a signal that the thread may continue
        if (send_signal) {
            pthread_cond_broadcast(&out_cond);
        }
        pthread_mutex_unlock(&out_mutex);
    }
}

void create_filter(FilterArgs *filter_args) {
    pthread_t *filter_tid = (pthread_t *)malloc(sizeof(pthread_t));
    pthread_create(filter_tid, NULL, filter, (void *)filter_args);
}

void *filter(void* pargs) {
    FilterArgs* filter_args = (FilterArgs*) pargs;
    int has_created = 0;
    struct queue* inbound_queue = filter_args->inbound_queue;
    int prime = filter_args->associated_prime;
    pthread_mutex_t *in_mutex = filter_args->in_mutex; 
    pthread_cond_t *in_cond = filter_args->in_cond;

    // create a new queue on the stack
    struct queue outbound_queue;
    outbound_queue.first = NULL;
    outbound_queue.last = NULL;
    int value;

    pthread_mutex_t out_mutex;
    pthread_cond_t out_cond;
    pthread_mutex_init(&out_mutex, NULL);
    pthread_cond_init(&out_cond, NULL);

    while (1) {
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
            // Wait for inbound to have a value again
            pthread_cond_wait(in_cond, in_mutex);
            pthread_mutex_unlock(in_mutex);
            continue;
        }

        // If the number is dividable, discard
        if (value % prime == 0)
            continue;
        else {
            // Check if the filter is at the end of the chain
            if (!has_created) {
                FilterArgs new_filter_args;
                new_filter_args.associated_prime = value;
                new_filter_args.inbound_queue = &outbound_queue;
                new_filter_args.in_mutex = &out_mutex;
                new_filter_args.in_cond = &out_cond;

                printf("%i ", value);
                fflush(stdout);
                create_filter(&new_filter_args);
                has_created = 1;
            }
            // write to the outbound queue
            pthread_mutex_lock(&out_mutex);
            if (is_empty(&outbound_queue)) {
                push(&outbound_queue, value);
                pthread_cond_broadcast(&out_cond);
            }
            else {
                push(&outbound_queue, value);
            }
            pthread_mutex_unlock(&out_mutex);
        }
    }
}
 