#include <stdlib.h>

struct entry {
    int value;
    struct entry *next;
};

int push(struct entry *queue_start, int value) {
    struct entry *new_entry = (struct entry*)malloc(sizeof(struct entry));
    new_entry->value = value;
    new_entry->next = queue_start;
    queue_start = new_entry;
    return 0;
}

int pop(struct entry *queue_start) {
    struct entry *current_entry = queue_start;
    struct entry *previous_entry = NULL;

    while (current_entry->next != NULL) {
        previous_entry = current_entry;
        current_entry = current_entry->next;
    }
    int value = current_entry->value;
    free(current_entry);
    previous_entry->next = NULL;
    return value;
}