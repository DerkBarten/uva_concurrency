#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "queue.c"


/*
TODO's:

- The inboundqueue GeneratorThread and the FilterThread shouldn't be ints but should be 'queue'-structs,
- Also change every placeholder throughout the code where the incorrect object is used.
- Add an boolean to the FilterThread which indicates if it has already created a new thread
- Threads in main are now created with pid's, but don't know if this wat is correct 
- Set the inbound-queue of the First FilterThread to the outbound-queue of the Generator Thread

- Check if the generate number function works


- This whole beast should be made threadsafe

*/

typedef struct GeneratorThread{
    int start_value;
    // change to correct struct
    // this linked list is also the outbound queue
    int outbound_queue;
} GeneratorThread;

typedef struct FilterThread{
    // pointer to outputqueue
    int associated_prime = 0
    // every thread needs a pointer to the inboundqueu
    int inbound_queue 
} FilterThread;


int main(int argc, char *argv[]){

    // change later, such that each filterthread also gets an ID
    pthread_t tid[10];
    // create thread for generateor
    GeneratorThread *generator = (GeneratorThread *)malloc(sizeof(GeneratorThread));
    generator->start_value = 2;
    generator->linked_list = 0;

    // creat generateor thread, what should ID be
    pthread_create(&tid[0], NULL, generate_numbers, (void *)args);

    // create the first FilterThread whith associated_value:2
    // should point to the queueu of the generator thread
    FilterThread *filter = (FilterThread *)malloc(sizeof(FilterThread));
    filter->associated_prime = 2;
    filter->inbound = generator->outbound_queue;
    // set the associated_prime of each number
    // set the inbound_queue of this thread
    pthread_create(&tid[1], NULL, filterNumbers, (void *)args);
}

// this function will generate the functions for the generateor-thread
void* generate_numbers(void* pargs) {
    GeneratorThread* generator = (GeneratorThread*) pargs;
    while(true) {
        push(linkedlist, startvalue)
        generator->start_value += 1
    }
}

// this function will generate the functions for the generateor-thread
void* filter_numbers(void* pargs) {
    FilterThread* filter = (FilterThread*) pargs;
    int has_created = 0
    int x = filter.associated_prime;
    
    // create  the outboundqueue
    struct queue *outbound_queue = malloc(sizeof(struct queue))

    while(!isEmpty(outbound_queue)) {
        int value = pop(outbound_queue)
        if(value % x == 0) continue;
        else {
            if (!has_created) {
                FilterThread *args = (FilterThread *)malloc(sizeof(FilterThread));
                args->associated_prime = value;
                args->inbound = outbound_queue;
                // create a valid thread_id
                pthread_create(&tid[1], NULL, filterNumbers, (void *)args);
                has_created = 1;
            } else {
                // write to the outboundqueue
                push(outbound_queue, value)
            }
        }
    }
}
 