CC=gcc
CFLAGS=-O2 -fopenmp -Wall -Wextra -std=c11

all: fib_tasks

fib_tasks: fib_tasks.c
	$(CC) $(CFLAGS) fib_tasks.c -o fib_tasks

run: fib_tasks
	./fib_tasks 40 20 4

clean:
	rm -f fib_tasks
