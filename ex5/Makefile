CC=gcc
CFLAGS=-fopenmp -O2

all: pi_seq pi_par pi_par_critical

pi_seq: pi_seq.c
	$(CC) $(CFLAGS) pi_seq.c -o pi_seq

pi_par: pi_par.c
	$(CC) $(CFLAGS) pi_par.c -o pi_par

pi_par_critical: pi_par_critical.c
	$(CC) $(CFLAGS) pi_par_critical.c -o pi_par_critical

run: pi_seq pi_par
	./pi_seq
	./pi_par

run_comparison: pi_par_critical
	./pi_par_critical

run_all: all
	@echo "=== Sequential Version ==="
	./pi_seq 100000000
	@echo "\n=== Parallel Version (Task 1) ==="
	./pi_par 100000000
	@echo "\n=== Performance Comparison (Task 2) ==="
	./pi_par_critical 100000000

clean:
	rm -f pi_seq pi_par pi_par_critical