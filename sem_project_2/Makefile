CC      = mpicc
CFLAGS  = -O3 -Wall -Wextra
LDFLAGS = -lm

all: matmul matmul_v2

matmul: matmul.c
	$(CC) $(CFLAGS) -o matmul matmul.c $(LDFLAGS)

matmul_v2: matmul_v2.c
	$(CC) $(CFLAGS) -o matmul_v2 matmul_v2.c $(LDFLAGS)

test: all
	mpirun -np 4 ./matmul 4 42 1
	@echo "---"
	mpirun -np 4 ./matmul_v2 4 42 1

verify: all
	mpirun -np 4 ./matmul 4 42 0
	mpirun -np 4 ./matmul_v2 4 42 0

clean:
	rm -f matmul matmul_v2

.PHONY: all test verify clean
