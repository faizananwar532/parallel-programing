CC=gcc
CFLAGS=-fopenmp

all: pi_seq pi_par

pi_seq: pi_seq.c
	$(CC) $(CFLAGS) pi_seq.c -o pi_seq

pi_par: pi_par.c
	$(CC) $(CFLAGS) pi_par.c -o pi_par

run: pi_seq pi_par
	./pi_seq
	./pi_par

clean:
	rm -f pi_seq pi_par