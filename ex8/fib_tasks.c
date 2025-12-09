#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static long long fib_seq(int n) {
    if (n < 2) {
        return n;
    }
    return fib_seq(n - 1) + fib_seq(n - 2);
}

// Task-based Fibonacci with cutoff to limit task overhead
static long long fib_task_impl(int n, int cutoff) {
    if (n < 2) {
        return n;
    }
    if (n <= cutoff) {
        return fib_seq(n);
    }

    long long x = 0;
    long long y = 0;

    // Spawn child tasks; use firstprivate to pass n and cutoff by value
    #pragma omp task shared(x) firstprivate(n, cutoff)
    {
        x = fib_task_impl(n - 1, cutoff);
    }

    #pragma omp task shared(y) firstprivate(n, cutoff)
    {
        y = fib_task_impl(n - 2, cutoff);
    }

    #pragma omp taskwait
    return x + y;
}

static long long fib_task(int n, int cutoff) {
    long long result = 0;
    #pragma omp parallel
    {
        #pragma omp single
        {
            result = fib_task_impl(n, cutoff);
        }
    }
    return result;
}

static void verify_small_cases(void) {
    int ok = 1;
    for (int i = 0; i <= 15; i++) {
        long long s = fib_seq(i);
        long long p = fib_task(i, 10); // small cutoff for tiny n
        if (s != p) {
            fprintf(stderr, "Mismatch at n=%d: seq=%lld, task=%lld\n", i, s, p);
            ok = 0;
        }
    }
    if (ok) {
        printf("[check] Small-n correctness verified (n=0..15).\n\n");
    }
}

int main(int argc, char **argv) {
    int n = 40;
    int cutoff = 20;
    int threads = 4;

    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (argc > 2) {
        cutoff = atoi(argv[2]);
    }
    if (argc > 3) {
        threads = atoi(argv[3]);
    }

    if (n < 0 || n > 92) {
        fprintf(stderr, "n must be between 0 and 92 (fits in signed 64-bit).\n");
        return 1;
    }
    if (cutoff < 0) {
        fprintf(stderr, "cutoff must be non-negative.\n");
        return 1;
    }
    if (threads < 1) {
        fprintf(stderr, "threads must be >= 1.\n");
        return 1;
    }

    omp_set_dynamic(0);
    omp_set_num_threads(threads);

    printf("n=%d, cutoff=%d, threads=%d\n", n, cutoff, threads);

    // Verify small cases once
    verify_small_cases();

    double t0 = omp_get_wtime();
    long long seq_result = fib_seq(n);
    double t1 = omp_get_wtime();

    double t2 = omp_get_wtime();
    long long par_result = fib_task(n, cutoff);
    double t3 = omp_get_wtime();

    printf("Sequential fib(%d) = %lld, time = %.6f s\n", n, seq_result, t1 - t0);
    printf("Tasked    fib(%d) = %lld, time = %.6f s\n", n, par_result, t3 - t2);
    if (par_result != seq_result) {
        printf("[warn] Results differ!\n");
    }

    if (t3 - t2 > 0.0) {
        printf("Speedup vs sequential: %.2fx\n", (t1 - t0) / (t3 - t2));
    }

    printf("\nNotes: cutoff controls task creation. Too small -> overhead; too large -> less parallelism.\n");
    return 0;
}
