CC=gcc
SRC_FILES=main.c image.c
OBJ_FILES=main.o image.o

LINKER_FLAGS=-lm

PROGRAM_NAME=main
RUNARGS = kitten.jpg output.png
.PHONY: all clean

run: $(PROGRAM_NAME)
	./$(PROGRAM_NAME) $(RUNARGS)
	display output.png

all: $(OBJ_FILES) $(PROGRAM_NAME)

%.o: %.c
	$(CC) -c $<

$(PROGRAM_NAME): $(OBJ_FILES)
	$(CC)  $^ $(LINKER_FLAGS) -o $@

clean:
	rm -rf *.o $(PROGRAM_NAME)