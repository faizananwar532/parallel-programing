/*
 * matmul_v2.c — Parallel MPI Matrix Multiplication (Strategy 2)
 *
 * Parallelization: Inner-dimension (k) split with MPI_Reduce
 *   - Each process locally initializes only the data it needs using the
 *     deterministic RNG (same my_rand/concatenate from row_wise_matrix_mult.c)
 *     → eliminates all initialization communication
 *   - Each process owns a slice of the k-dimension: k in [k_start, k_end)
 *   - Each process computes a partial n×n result matrix C_partial
 *   - All partial results are summed at rank 0 via MPI_Reduce with MPI_SUM
 *   - Timing via MPI_Wtime (Lecture 10)
 *
 * Key difference from Strategy 1:
 *   Strategy 1 splits ROWS of A → each process gets complete rows of C
 *   Strategy 2 splits the INNER DIMENSION k → each process gets a partial C
 *
 * MPI concepts used (all from lecture slides):
 *   Lecture 09: MPI_Init, MPI_Finalize, MPI_Comm_rank, MPI_Comm_size,
 *              MPI_COMM_WORLD, MPI_DOUBLE
 *   Lecture 10: MPI_Wtime
 *   Lecture 11: MPI_Reduce, MPI_SUM
 *
 * Usage: mpirun -np <P> ./matmul_v2 n seed verbose
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

/* ---- Element-wise matrix generation using the deterministic RNG ---- */
/* Since my_rand depends only on (i, j, seed, value), any process can
   independently compute any matrix element without communication. */

static inline double gen_element(int row, int col, int seed, int value)
{
    unsigned long state = concatenate(row, col) + seed + value;
    return my_rand(&state, 0, 1);
}

/* Fill full n×n matrix into contiguous buffer (for printing) */
void fillMatrix(double *arr, int n, int seed, int value)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            arr[i * n + j] = gen_element(i, j, seed, value);
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

    /* Parse command-line arguments */
    if (argc < 4) {
        if (rank == 0)
            fprintf(stderr, "Usage: mpirun -np <P> ./matmul_v2 n seed verbose\n");
        MPI_Finalize();
        return 1;
    }

    int n       = atoi(argv[1]);
    int seed    = atoi(argv[2]);
    int verbose = atoi(argv[3]);

    /* Start timing immediately after reading parameters (as required) */
    double t_start = MPI_Wtime();

    /* ---- Compute k-dimension distribution ---- */
    /* Each process gets a slice of the inner summation index k */
    int base_k    = n / size;
    int remainder = n % size;
    int my_k      = base_k + (rank < remainder ? 1 : 0);
    int k_start   = rank * base_k + (rank < remainder ? rank : remainder);

    /* ---- Allocate memory ---- */
    /* A_cols: n rows × my_k columns (the columns of A this process owns) */
    double *A_cols = (double *)malloc((size_t)n * my_k * sizeof(double));
    /* B_rows: my_k rows × n columns (the rows of B this process owns) */
    double *B_rows = (double *)malloc((size_t)my_k * n * sizeof(double));
    /* C_partial: n×n partial result (will be reduced via MPI_Reduce) */
    double *C_partial = (double *)calloc((size_t)n * n, sizeof(double));
    /* C_result: n×n final result, only needed at rank 0 */
    double *C_result = NULL;
    if (rank == 0) {
        C_result = (double *)malloc((size_t)n * n * sizeof(double));
    }

    /* ---- Initialize locally — no communication needed ---- */
    /* Each process generates only the matrix elements it needs.
       This works because my_rand is a pure function of (i, j, seed, value). */

    /* Fill columns [k_start, k_start+my_k) of A for all n rows */
    for (int i = 0; i < n; i++) {
        for (int c = 0; c < my_k; c++) {
            A_cols[i * my_k + c] = gen_element(i, k_start + c, seed, 0);
        }
    }

    /* Fill rows [k_start, k_start+my_k) of B for all n columns */
    for (int r = 0; r < my_k; r++) {
        for (int j = 0; j < n; j++) {
            B_rows[r * n + j] = gen_element(k_start + r, j, seed, 1);
        }
    }

    /* ---- Matrix multiplication: partial outer product ---- */
    /* C_partial[i][j] += sum over local k of A[i][k] * B[k][j]
       ikj loop order: inner j-loop accesses C_partial and B_rows
       sequentially in memory → cache-friendly */
    for (int i = 0; i < n; i++) {
        for (int kk = 0; kk < my_k; kk++) {
            double a_val = A_cols[i * my_k + kk];
            for (int j = 0; j < n; j++) {
                C_partial[i * n + j] += a_val * B_rows[kk * n + j];
            }
        }
    }

    /* ---- Reduce all partial C matrices to rank 0 (MPI_Reduce, Lecture 11) ---- */
    /* MPI_SUM combines all partial results element-wise at root */
    MPI_Reduce(C_partial, C_result, n * n, MPI_DOUBLE, MPI_SUM,
               0, MPI_COMM_WORLD);

    /* ---- Output (rank 0 only) ---- */
    /* Note: rank 0 now contains the complete matrix C in C_result */
    if (rank == 0) {
        /* Print matrices if verbose=1 and n <= 10 */
        if (verbose == 1 && n <= 10) {
            double *A_full = (double *)malloc((size_t)n * n * sizeof(double));
            double *B_full = (double *)malloc((size_t)n * n * sizeof(double));
            fillMatrix(A_full, n, seed, 0);
            fillMatrix(B_full, n, seed, 1);
            printMatrix("Matrix A:", A_full, n);
            printMatrix("Matrix B:", B_full, n);
            printMatrix("Matrix C (Result):", C_result, n);
            free(A_full);
            free(B_full);
        }

        /* Compute checksum = sum of all C[i][j] */
        double checksum = 0.0;
        for (int i = 0; i < n * n; i++) {
            checksum += C_result[i];
        }
        printf("Checksum: %f\n", checksum);
    }

    /* Stop timing immediately before MPI_Finalize */
    double t_end = MPI_Wtime();

    if (rank == 0) {
        printf("Execution time with %d ranks: %.2f s\n", size, t_end - t_start);
    }

    /* ---- Cleanup ---- */
    free(A_cols);
    free(B_rows);
    free(C_partial);
    if (rank == 0) {
        free(C_result);
    }

    MPI_Finalize();
    return 0;
}
