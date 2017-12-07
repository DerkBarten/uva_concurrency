PROGNAME = main
#SRCFILES = main.cpp file.cpp simulate.cpp
#TARNAME = assign1_2.tgz

RUNARGS = 1000 1000 512 # i_max t_max threadBlockSize, increase this when testing on the DAS4!

IMAGEVIEW = display
NVCC = nvcc
CC = g++

CU_FLAGS	= -O3 -g --ptxas-options=-v -arch compute_20 -code sm_20
CC_FLAGS	= -O3 -m64 -Wall

#WARNFLAGS = -Wall -Werror-implicit-function-declaration -Wshadow \
		  -Wstrict-prototypes -pedantic-errors
#CFLAGS = -std=c99 -ggdb -O2 $(WARNFLAGS) -D_POSIX_C_SOURCE=200112 -fopenmp
#LFLAGS = -lm -lrt

CU_MAIN	= main.cu
CU_SOURCES = simulate.cu
CC_SOURCES = file.cc generatedata.cc

# Do some substitution to get a list of .o files from the given .c files.
CU_OBJECTS	= $(CU_SOURCES:%.cu=%.o)
CU_PTX		= $(CU_SOURCES:%.cu=%.ptx)
CC_OBJECTS	= $(CC_SOURCES:%.cc=%.o)


.PHONY: all run runlocal plot clean dist todo

all: $(PROGNAME)

$(PROGNAME): $(CU_MAIN) $(CU_OBJECTS) $(CC_OBJECTS)
	$(NVCC) $^ -o $@

%.o: %.cu
	$(NVCC) $(CU_FLAGS) -c $< -o $@

%.o: %.cc
	$(CC) $(CC_FLAGS) -c $< -o $@

%.ptx: %.cu
	$(NVCC) $(CU_FLAGS) --ptx $< -o $@

ptx: $(CU_PTX) 

run: $(PROGNAME)
	prun -v -np 1 -native '-l gpu=GTX480' ./$(PROGNAME) $(RUNARGS)

runlocal: $(PROGNAME)
	./$(PROGNAME) $(RUNARGS)

plot: result.txt
	gnuplot plot.gnp
	$(IMAGEVIEW) plot.png

#dist:
#	tar cvzf $(TARNAME) Makefile *.c *.h data/

clean:
	rm -fv $(PROGNAME) *.o $(TARNAME) result.txt plot.png vector-add



#CU_FLAGS	= -O3 -g -deviceemu








