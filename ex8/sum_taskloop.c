#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double *alloc_array(long n) {
    double *a = (double *)malloc((size_t)n * sizeof(double));
    if (!a) {
        fprintf(stderr, "Allocation failed for n=%ld\n", n);
        exit(1);
    }
    return a;
}

static void fill_random(double *a, long n) {
    // Use a fixed seed for reproducibility; change to time(NULL) if desired
    srand(42);
    for (long i = 0; i < n; i++) {
        a[i] = (double)rand() / (double)RAND_MAX; // in [0,1]
    }
}

static double sum_parallel_for(const double *a, long n, int threads) {
    double sum = 0.0;
    omp_set_num_threads(threads);
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < n; i++) {
        sum += a[i];
    }
    return sum;
}

static double sum_taskloop(const double *a, long n, int threads, long grainsize, long num_tasks) {
    double sum = 0.0;
    omp_set_num_threads(threads);
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            if (grainsize > 0) {
                #pragma omp taskloop reduction(+:sum) grainsize(grainsize)
                for (long i = 0; i < n; i++) {
                    sum += a[i];
                }
            } else if (num_tasks > 0) {
                #pragma omp taskloop reduction(+:sum) num_tasks(num_tasks)
                for (long i = 0; i < n; i++) {
                    sum += a[i];
                }
            } else {
                #pragma omp taskloop reduction(+:sum)
                for (long i = 0; i < n; i++) {
                    sum += a[i];
                }
            }
        }
    }
    return sum;
}

int main(int argc, char **argv) {
    // Args: n threads grainsize num_tasks
    long n = 50 * 1000 * 1000L; // 50 million elements ~400 MB
    int threads = 8;
    long grainsize = 0; // default: let OpenMP choose
    long num_tasks = 0; // default: let OpenMP choose

    if (argc > 1) n = atol(argv[1]);
    if (argc > 2) threads = atoi(argv[2]);
    if (argc > 3) grainsize = atol(argv[3]);
    if (argc > 4) num_tasks = atol(argv[4]);

    if (n <= 0) {
        fprintf(stderr, "n must be positive.\n");
        return 1;
    }
    if (threads <= 0) {
        fprintf(stderr, "threads must be positive.\n");
        return 1;
    }

    printf("n=%ld, threads=%d, grainsize=%ld, num_tasks=%ld\n", n, threads, grainsize, num_tasks);

    double *a = alloc_array(n);
    fill_random(a, n);

    // Parallel for baseline
    double t0 = omp_get_wtime();
    double sum_for = sum_parallel_for(a, n, threads);
    double t1 = omp_get_wtime();

    // Taskloop variant
    double t2 = omp_get_wtime();
    double sum_task = sum_taskloop(a, n, threads, grainsize, num_tasks);
    double t3 = omp_get_wtime();

    printf("parallel for : sum=%.6f time=%.6f s\n", sum_for, t1 - t0);
    printf("taskloop     : sum=%.6f time=%.6f s\n", sum_task, t3 - t2);
    if (t3 - t2 > 0.0) {
        printf("speedup (taskloop vs for): %.3fx\n", (t1 - t0) / (t3 - t2));
    }
    if (llabs((long long)(sum_for - sum_task)) > 1e-6) {
        printf("[warn] sums differ (possible FP drift).\n");
    }

    free(a);

    printf("\nGuidance:\n");
    printf("- taskloop is useful when work per iteration is imbalanced or irregular.\n");
    printf("- For regular dense loops like this sum, parallel for is usually faster due to lower overhead.\n");
    printf("- grainsize or num_tasks lets you tune task granularity; too small -> overhead, too large -> underutilization.\n");

    return 0;
}
