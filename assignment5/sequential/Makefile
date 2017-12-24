CC=gcc
NODES=5

SRC_FILES=main.c image.c timer.c
OBJ_FILES=main.o image.o timer.o

LINKER_FLAGS=-lm -lrt
C_FLAGS=-std=c99

PROGRAM_NAME=main
RUNARGS=../images/kitten.jpg output.png

.PHONY: all clean

all: $(OBJ_FILES) $(PROGRAM_NAME)

#timer.o: timer.c
#	$(CC) -fopenmp -c timer.c
 
%.o: %.c
	$(CC) $(C_FLAGS) -c $<

$(PROGRAM_NAME): $(OBJ_FILES)
	$(CC) $(LINKER_FLAGS) $^ -o $@

run: $(PROGRAM_NAME)
	prun -v -np $(NODES) ./$(PROGRAM_NAME) $(RUNARGS)

benchmark: $(PROGRAM_NAME)
	prun -v -np $(NODES) time ./$(PROGRAM_NAME) $(RUNARGS)

show:
	display output.png

clean:
	rm -rf *.o $(PROGRAM_NAME) output.png