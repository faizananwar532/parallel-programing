/*
 * Parallel Matrix Multiplication using MPI
 * Strategy: Row-wise block distribution with cache-optimized tiled multiplication
 *
 * Each process:
 *   1. Initializes its own rows of A locally (no scatter needed)
 *   2. Initializes the full B matrix locally (deterministic, no broadcast needed)
 *   3. Computes its rows of C using tiled i,k,j loop order
 *   4. Sends its rows of C to rank 0 via MPI_Gatherv
 *
 * Author: [Your Name]
 * AI tools were used as a reference during development (Cursor AI).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define BLOCK_SIZE 64

static int MATRIX_SIZE;
static int SEED;

unsigned concatenate(unsigned x, unsigned y) {
    unsigned pow = 10;
    while (y >= pow)
        pow *= 10;
    return x * pow + y;
}

double my_rand(unsigned long *state, double lower, double upper) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    unsigned long x = (*state * 0x2545F4914F6CDD1DULL);
    const double inv = 1.0 / (double)(1ULL << 53);
    double u = (double)(x >> 11) * inv;
    return lower + (upper - lower) * u;
}

static void fill_local_rows(double *arr, int row_start, int num_rows, int value) {
    for (int i = 0; i < num_rows; i++) {
        int global_i = row_start + i;
        for (int j = 0; j < MATRIX_SIZE; j++) {
            unsigned long state = concatenate(global_i, j) + SEED + value;
            arr[i * MATRIX_SIZE + j] = my_rand(&state, 0, 1);
        }
    }
}

static void fill_matrix(double *arr, int value) {
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            unsigned long state = concatenate(i, j) + SEED + value;
            arr[i * MATRIX_SIZE + j] = my_rand(&state, 0, 1);
        }
    }
}

static void print_matrix(const char *name, double *mat, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (j > 0) printf(" ");
            printf("%f", mat[i * cols + j]);
        }
        printf("\n");
    }
}

/*
 * Tiled matrix multiplication kernel: C += A * B
 * Uses i,k,j loop order with blocking for cache efficiency.
 * A is (my_rows x n), B is (n x n), C is (my_rows x n).
 */
static void matmul_tiled(double *A, double *B, double *C, int my_rows, int n) {
    for (int ii = 0; ii < my_rows; ii += BLOCK_SIZE) {
        int i_end = ii + BLOCK_SIZE;
        if (i_end > my_rows) i_end = my_rows;

        for (int kk = 0; kk < n; kk += BLOCK_SIZE) {
            int k_end = kk + BLOCK_SIZE;
            if (k_end > n) k_end = n;

            for (int jj = 0; jj < n; jj += BLOCK_SIZE) {
                int j_end = jj + BLOCK_SIZE;
                if (j_end > n) j_end = n;

                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        double a_ik = A[i * n + k];
                        for (int j = jj; j < j_end; j++) {
                            C[i * n + j] += a_ik * B[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n       = (argc > 1) ? atoi(argv[1]) : 8000;
    int seed    = (argc > 2) ? atoi(argv[2]) : 42;
    int verbose = (argc > 3) ? atoi(argv[3]) : 0;

    SEED = seed;
    MATRIX_SIZE = n;

    double start_time = MPI_Wtime();

    /* --- Row distribution ------------------------------------------------ */
    int base_rows = n / size;
    int remainder = n % size;
    int my_rows = base_rows + (rank < remainder ? 1 : 0);
    int my_start = rank * base_rows + (rank < remainder ? rank : remainder);

    /* --- Allocate memory ------------------------------------------------- */
    double *local_A = (double *)malloc((size_t)my_rows * n * sizeof(double));
    double *B       = (double *)malloc((size_t)n * n * sizeof(double));
    double *local_C = (double *)calloc((size_t)my_rows * n, sizeof(double));

    if (!local_A || !B || !local_C) {
        fprintf(stderr, "Rank %d: memory allocation failed\n", rank);
        MPI_Finalize();
        return 1;
    }

    /* --- Initialize matrices locally (no communication!) ----------------- */
    fill_local_rows(local_A, my_start, my_rows, 0);
    fill_matrix(B, 1);

    /* --- Tiled matrix multiplication ------------------------------------- */
    matmul_tiled(local_A, B, local_C, my_rows, n);

    /* --- Gather C on rank 0 ---------------------------------------------- */
    double *C = NULL;
    int *recvcounts = NULL;
    int *displs = NULL;

    if (rank == 0) {
        C = (double *)malloc((size_t)n * n * sizeof(double));
        recvcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));

        int offset = 0;
        for (int r = 0; r < size; r++) {
            int r_rows = base_rows + (r < remainder ? 1 : 0);
            recvcounts[r] = r_rows * n;
            displs[r] = offset;
            offset += r_rows * n;
        }
    }

    MPI_Gatherv(local_C, my_rows * n, MPI_DOUBLE,
                C, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* --- Output (rank 0 only) -------------------------------------------- */
    if (rank == 0) {
        if (verbose == 1 && n <= 10) {
            double *full_A = (double *)malloc((size_t)n * n * sizeof(double));
            fill_matrix(full_A, 0);
            print_matrix("Matrix A", full_A, n, n);
            free(full_A);

            print_matrix("Matrix B", B, n, n);
            print_matrix("Matrix C (Result)", C, n, n);
        }

        double checksum = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                checksum += C[i * n + j];

        printf("Checksum: %f\n", checksum);
    }

    double end_time = MPI_Wtime();

    if (rank == 0)
        printf("Execution time with %d ranks: %.2f s\n", size, end_time - start_time);

    /* --- Cleanup --------------------------------------------------------- */
    free(local_A);
    free(B);
    free(local_C);
    if (rank == 0) {
        free(C);
        free(recvcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}
