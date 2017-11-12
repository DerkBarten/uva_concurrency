
all:
	gcc -c queue_test.c 
	gcc -c queue.c
	gcc queue_test.o queue.o -o eratosthenes

clean:
	rm -fv $(PROGNAME) $(OBJFILES) $(TARNAME) result.txt plot.png
