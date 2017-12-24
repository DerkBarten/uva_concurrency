CC=gcc
NODES=1

SRC_FILES=main.c image.c timer.c
OBJ_FILES=main.o image.o timer.o

LINKER_FLAGS=-lm
MPFLAGS = -fopenmp

PROGRAM_NAME=main
RUNARGS = ../images/kitten.jpg output.png

.PHONY: all clean run show benchmark

all: $(OBJ_FILES) $(PROGRAM_NAME)

runlocal: $(PROGRAM_NAME)
	./$(PROGRAM_NAME) $(RUNARGS) 

%.o: %.c
	$(CC) $(MPFLAGS) -c $<

$(PROGRAM_NAME): $(OBJ_FILES)
	$(CC) $^ $(LINKER_FLAGS) $(MPFLAGS)  -o  $@

run: $(PROGRAM_NAME)
	prun -v -np $(NODES) ./$(PROGRAM_NAME) $(RUNARGS)

benchmark: $(PROGRAM_NAME)
	prun -v -np $(NODES) time ./$(PROGRAM_NAME) $(RUNARGS)

show:
	display output.png

clean:
	rm -rf *.o $(PROGRAM_NAME) output.png