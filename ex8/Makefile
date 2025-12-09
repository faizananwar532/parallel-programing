CC=gcc
CFLAGS=-O2 -fopenmp -Wall -Wextra -std=c11

all: fib_tasks sum_taskloop

fib_tasks: fib_tasks.c
	$(CC) $(CFLAGS) fib_tasks.c -o fib_tasks

sum_taskloop: sum_taskloop.c
	$(CC) $(CFLAGS) sum_taskloop.c -o sum_taskloop

run_fib: fib_tasks
	./fib_tasks 40 20 4

run_sum: sum_taskloop
	./sum_taskloop 50000000 8 0 0

clean:
	rm -f fib_tasks sum_taskloop
