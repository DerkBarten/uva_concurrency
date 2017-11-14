CC = gcc

WARNFLAGS = -Wall -Werror-implicit-function-declaration -Wshadow \
		  -Wstrict-prototypes -pedantic-errors
CFLAGS = -std=c99 -ggdb -O2 $(WARNFLAGS) -D_POSIX_C_SOURCE=200112
LFLAGS = -lm -lrt -lpthread

PROGNAME=eratosthenes
SRCFILES=$(wildcard *.c)
OBJFILES=$(wildcard *.o)

all:
	gcc -c $(CFLAGS) queue_test.c 
	gcc -c $(CFLAGS) queue.c
	gcc queue_test.o queue.o -o queuetest
	gcc -c $(CFLAGS) main.c 
	gcc $(LFLAGS) main.o queue.o -o $(PROGNAME)

clean:
	rm -fv $(PROGNAME) $(OBJFILES) result.txt plot.png
