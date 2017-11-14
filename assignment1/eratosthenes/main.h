#include <pthread.h>
#include "queue.h"

typedef struct FilterArgs{
    int associated_prime;
    // every thread needs a pointer to the inboundqueu
    struct queue* inbound_queue;
    pthread_mutex_t *in_mutex;
    pthread_cond_t *in_cond;

} FilterArgs;

void create_filter(FilterArgs *filter_args);

void *filter(void* pargs);