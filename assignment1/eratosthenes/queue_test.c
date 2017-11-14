#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "queue.h"

int main(int argc, char *argv[]){
    struct queue queue;
    queue.first = NULL;
    queue.last = NULL;
    
    push(&queue, 12);
    push(&queue, 13);
    push(&queue, 14);

    //pop(&queue);
    pop(&queue);

    push(&queue, 15);

    pop(&queue);
    pop(&queue);

    print_queue(&queue);

    
}