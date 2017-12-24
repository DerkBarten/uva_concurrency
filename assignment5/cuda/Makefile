CC=gcc
NVCC = nvcc
NODES=1

C_SRC_FILES=main.c image.c timer.c
C_OBJ_FILES=main.o image.o timer.o

CU_SRC_FILES=cuda_image.cu
CU_OBJ_FILES=cuda_image.o

#CU_FLAGS=-ccbin clang-3.8 
CU_FLAGS=-arch compute_20 -code sm_20
C_FLAGS=

LINKER_FLAGS=-lm -lcuda

PROGRAM_NAME=main
RUNARGS=../images/kitten.jpg output.png

.PHONY: all clean run show benchmark

all: $(C_OBJ_FILES) $(CU_OBJ_FILES) $(PROGRAM_NAME)

%.o: %.c
	$(CC) $(C_FLAGS) -c $<

%.o: %.cu
	$(NVCC) $(CU_FLAGS) -c $< -o $@

$(PROGRAM_NAME): $(C_OBJ_FILES) $(CU_OBJ_FILES)
	$(NVCC) $(CU_FLAGS) $(LINKER_FLAGS) $^ -o $@

run: $(PROGRAM_NAME)
	prun -v -np $(NODES) -native '-l gpu=GTX480' ./$(PROGRAM_NAME) $(RUNARGS)

benchmark: $(PROGRAM_NAME)
	prun -v -np $(NODES) -native '-l gpu=GTX480' time ./$(PROGRAM_NAME) $(RUNARGS)

show:
	display output.png

clean:
	rm -rf *.o $(PROGRAM_NAME) output.png