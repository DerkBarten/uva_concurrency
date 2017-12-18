CC=gcc
NVCC = nvcc

C_SRC_FILES=main.c image.c
C_OBJ_FILES=main.o image.o

CU_SRC_FILES=cuda_image.cu
CU_OBJ_FILES=cuda_image.o

CU_FLAGS=-ccbin clang-3.8

LINKER_FLAGS=-lm -lcuda

PROGRAM_NAME=main

.PHONY: all clean

all: $(C_OBJ_FILES) $(CU_OBJ_FILES) $(PROGRAM_NAME)

%.o: %.c
	$(CC) $(LINKER_FLAGS) -c $<

%.o: %.cu
	$(NVCC) $(CU_FLAGS) -c $< -o $@

$(PROGRAM_NAME): $(C_OBJ_FILES) $(CU_OBJ_FILES)
	$(NVCC) $(CU_FLAGS) $(LINKER_FLAGS) $^ -o $@

clean:
	rm -rf *.o $(PROGRAM_NAME)