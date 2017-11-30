\debug		:= 0
include		./Makefile.inc

NVCC= nvcc
CC = g++
PROGNAME = encrypt

all: $(PROGNAME)

$(PROGNAME): encrypt.cu
	$(NVCC) -c encrypt.cu
	$(CC) -c timer.cc -o timer.o
	$(NVCC) -o $(PROJ_BASE)/encrypt encrypt.o timer.o 

run: $(PROGNAME)
	prun -v -np 1 -native '-l gpu=GTX480 ' ./$(PROGNAME)

clean:
	rm -f ./encrypt.o
	rm -f $(PROJ_BASE)/encrypt
