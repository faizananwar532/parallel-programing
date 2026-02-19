/*
 * matmul.c — Parallel MPI Matrix Multiplication (Strategy 1)
 *
 * Parallelization: Row-block distribution
 *   - Rank 0 initializes A and B using fillArray (from row_wise_matrix_mult.c)
 *   - B is broadcast to all processes via MPI_Bcast (Lecture 11)
 *   - Rows of A are distributed via MPI_Scatterv (Lecture 11)
 *   - Each process computes its block of rows of C
 *   - Result rows are collected at rank 0 via MPI_Gatherv (Lecture 11)
 *   - Timing via MPI_Wtime (Lecture 10)
 *
 * MPI concepts used (all from lecture slides):
 *   Lecture 09: MPI_Init, MPI_Finalize, MPI_Comm_rank, MPI_Comm_size,
 *              MPI_COMM_WORLD, MPI_DOUBLE
 *   Lecture 10: MPI_Wtime
 *   Lecture 11: MPI_Bcast, MPI_Scatterv, MPI_Gatherv
 *
 * Usage: mpirun -np <P> ./matmul n seed verbose
 *
 * Author: Faizan
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

/* ---- RNG functions from row_wise_matrix_mult.c ---- */

double my_rand(unsigned long *state, double lower, double upper)
{
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    unsigned long x = (*state * 0x2545F4914F6CDD1DULL);
    const double inv = 1.0 / (double)(1ULL << 53);
    double u = (double)(x >> 11) * inv;
    return lower + (upper - lower) * u;
}

unsigned concatenate(unsigned x, unsigned y)
{
    unsigned pow = 10;
    while (y >= pow)
        pow *= 10;
    return x * pow + y;
}

/* ---- Matrix initialization (adapted from row_wise_matrix_mult.c) ---- */
/* Uses contiguous 1D array (double*) instead of double** for MPI compatibility */

void fillMatrix(double *arr, int n, int seed, int value)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            unsigned long state = concatenate(i, j) + seed + value;
            arr[i * n + j] = my_rand(&state, 0, 1);
        }
    }
}

void printMatrix(const char *label, double *arr, int n)
{
    printf("%s\n", label);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f", arr[i * n + j]);
            if (j < n - 1) printf(" ");
        }
        printf("\n");
    }
    printf("\n");
}

/* ---- Main ---- */

int main(int argc, char *argv[])
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Parse command-line arguments (with defaults for tool compatibility) */
    int n       = (argc > 1) ? atoi(argv[1]) : 8000;
    int seed    = (argc > 2) ? atoi(argv[2]) : 42;
    int verbose = (argc > 3) ? atoi(argv[3]) : 0;

    /* Start timing immediately after reading parameters (as required) */
    double t_start = MPI_Wtime();

    /* ---- Compute row distribution ---- */
    /* Distribute rows as evenly as possible; first 'remainder' ranks get +1 row */
    int base_rows = n / size;
    int remainder  = n % size;
    int my_rows    = base_rows + (rank < remainder ? 1 : 0);

    /* Build sendcounts and displacements for MPI_Scatterv / MPI_Gatherv */
    int *sendcounts = (int *)malloc(size * sizeof(int));
    int *displs     = (int *)malloc(size * sizeof(int));

    int offset = 0;
    for (int r = 0; r < size; r++) {
        int r_rows = base_rows + (r < remainder ? 1 : 0);
        sendcounts[r] = r_rows * n;   /* number of doubles for rank r */
        displs[r]     = offset;
        offset += sendcounts[r];
    }

    /* ---- Allocate memory ---- */
    /* Contiguous 1D arrays for MPI collective compatibility */
    double *A = NULL;   /* full A, only at rank 0 */
    double *B = (double *)malloc((size_t)n * n * sizeof(double));
    double *C = NULL;   /* full C, only at rank 0 */

    double *A_local = (double *)malloc((size_t)my_rows * n * sizeof(double));
    double *C_local = (double *)calloc((size_t)my_rows * n, sizeof(double));

    /* ---- Initialize matrices at rank 0 ---- */
    if (rank == 0) {
        A = (double *)malloc((size_t)n * n * sizeof(double));
        C = (double *)malloc((size_t)n * n * sizeof(double));
        fillMatrix(A, n, seed, 0);
        fillMatrix(B, n, seed, 1);
    }

    /* ---- Broadcast B to all processes (MPI_Bcast, Lecture 11) ---- */
    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ---- Distribute rows of A (MPI_Scatterv, Lecture 11) ---- */
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 A_local, my_rows * n, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    /* ---- Matrix multiplication: C_local = A_local x B ---- */
    /* ikj loop order: inner j-loop accesses C_local[i][j] and B[k][j]
       sequentially in memory → cache-friendly (row-major access) */
    for (int i = 0; i < my_rows; i++) {
        for (int k = 0; k < n; k++) {
            double a_ik = A_local[i * n + k];
            for (int j = 0; j < n; j++) {
                C_local[i * n + j] += a_ik * B[k * n + j];
            }
        }
    }

    /* ---- Gather result rows at rank 0 (MPI_Gatherv, Lecture 11) ---- */
    MPI_Gatherv(C_local, my_rows * n, MPI_DOUBLE,
                C, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* ---- Output (rank 0 only) ---- */
    if (rank == 0) {
        /* Print matrices if verbose=1 and n <= 10 */
        if (verbose == 1 && n <= 10) {
            printMatrix("Matrix A:", A, n);
            printMatrix("Matrix B:", B, n);
            printMatrix("Matrix C (Result):", C, n);
        }

        /* Compute checksum = sum of all C[i][j] */
        double checksum = 0.0;
        for (int i = 0; i < n * n; i++) {
            checksum += C[i];
        }
        printf("Checksum: %f\n", checksum);
    }

    /* Stop timing immediately before MPI_Finalize*/
    double t_end = MPI_Wtime();

    if (rank == 0) {
        printf("Execution time with %d ranks: %.2f s\n", size, t_end - t_start);
    }

    /* ---- Cleanup ---- */
    free(A_local);
    free(B);
    free(C_local);
    free(sendcounts);
    free(displs);
    if (rank == 0) {
        free(A);
        free(C);
    }

    MPI_Finalize();
    return 0;
}
