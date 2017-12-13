CC=gcc
SRC_FILES=main.c image.c
OBJ_FILES=main.o image.o

LINKER_FLAGS=-lm

PROGRAM_NAME=main

.PHONY: all clean

all: $(OBJ_FILES) $(PROGRAM_NAME)

%.o: %.c
	$(CC) $(LINKER_FLAGS) -c $<

$(PROGRAM_NAME): $(OBJ_FILES)
	$(CC) $(LINKER_FLAGS) $^ -o $@

clean:
	rm -rf *.o $(PROGRAM_NAME)