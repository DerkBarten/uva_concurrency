#include <stdlib.h>
#include <stdio.h>

#include "queue.h"

int push(struct queue *queue, int value) {
    struct queue_entry *new_entry = (struct queue_entry*)malloc(sizeof(struct queue_entry));
    new_entry->value = value;
    new_entry->next = queue->last;
    new_entry->previous = NULL;

    if (queue->first == NULL) {
        queue->first = new_entry;
    }

    if (queue->last != NULL) {
        queue->last->previous = new_entry;
    }

    queue->last = new_entry;
    return 0;
}

int pop(struct queue *queue) {
    if (queue->first == NULL) {
        return -1;
    }
    
    int value = queue->first->value;
    // If the length of the queue is one
    if (queue->first->previous == NULL) {
        free(queue->first);
        queue->first = NULL;
        queue->last = NULL;
    }
    else {
        struct queue_entry *new_first = queue->first->previous;
        new_first->next = NULL;
        
        free(queue->first);
        queue->first = new_first;
    }

    return value;
}

int print_queue(struct queue *queue){
    struct queue_entry *current_entry = queue->last;
    if (current_entry == NULL) {
        return -1;
    }

    while (current_entry->next != NULL) {
        printf("%i -> ", current_entry->value);
        current_entry = current_entry->next;
    }
    printf("%i\n", current_entry->value);
    return 0;
}

int is_empty(struct queue *queue) {
    if (queue->first == NULL) {
        return 1;
    }
    return 0;
}