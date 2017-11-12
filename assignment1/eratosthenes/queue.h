struct queue {
    struct queue_entry *first;
    struct queue_entry *last;
};

struct queue_entry {
    int value;
    struct queue_entry *next;
    struct queue_entry *previous;
};

int push(struct queue *queue, int value);

int pop(struct queue *queue);

int print_queue(struct queue *queue);

int is_empty(struct queue *queue);