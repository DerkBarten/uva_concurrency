CC=gcc
SRC_FILES=main.c image.c timer.c
OBJ_FILES=main.o image.o timer.o

LINKER_FLAGS=-lm
MPFLAGS = -fopenmp

PROGRAM_NAME=main
RUNARGS = hubble.jpg output.png
.PHONY: all clean

runlocal: $(PROGRAM_NAME)
	./$(PROGRAM_NAME) $(RUNARGS) 

run: $(PROGRAM_NAME)
	prun -v -np 1 ./$(PROGRAM_NAME) $(RUNARGS)

all: $(OBJ_FILES) $(PROGRAM_NAME)

%.o: %.c
	$(CC) $(MPFLAGS) -c $<

$(PROGRAM_NAME): $(OBJ_FILES)
	$(CC) $^ $(LINKER_FLAGS) $(MPFLAGS)  -o  $@

clean:
	rm -rf *.o $(PROGRAM_NAME)