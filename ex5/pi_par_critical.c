#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

double step;

// Version 1: Using critical section
double pi_critical(long num_steps, int num_threads) {
    double sum = 0.0;
    double x;
    long i;
    
    step = 1.0 / (double) num_steps;
    
    #pragma omp parallel for private(x) num_threads(num_threads)
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        #pragma omp critical
        {
            sum = sum + 4.0 / (1.0 + x * x);
        }
    }
    
    return step * sum;
}

// Version 2: Using atomic operation
double pi_atomic(long num_steps, int num_threads) {
    double sum = 0.0;
    double x;
    long i;
    
    step = 1.0 / (double) num_steps;
    
    #pragma omp parallel for private(x) num_threads(num_threads)
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        #pragma omp atomic
        sum += 4.0 / (1.0 + x * x);
    }
    
    return step * sum;
}

// Version 3: Using reduction clause (most efficient)
double pi_reduction(long num_steps, int num_threads) {
    double sum = 0.0;
    double x;
    long i;
    
    step = 1.0 / (double) num_steps;
    
    #pragma omp parallel for private(x) reduction(+:sum) num_threads(num_threads)
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    
    return step * sum;
}

// Version 4: Manual sum with local variables (like Task 1, but dynamic threads)
double pi_manual_sum(long num_steps, int num_threads) {
    double sum = 0.0;
    double x;
    long i;
    double *partial_sums;
    int actual_threads = 0;
    
    partial_sums = (double *)calloc(num_threads, sizeof(double));
    if (partial_sums == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 0.0;
    }
    
    step = 1.0 / (double) num_steps;
    
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        double partial_sum = 0.0;
        
        if (thread_id == 0) {
            actual_threads = total_threads;
        }
        
        // Block decomposition
        long start = (thread_id * num_steps) / total_threads;
        long end = ((thread_id + 1) * num_steps) / total_threads;
        
        for (i = start; i < end; i++) {
            x = (i + 0.5) * step;
            partial_sum = partial_sum + 4.0 / (1.0 + x * x);
        }
        
        partial_sums[thread_id] = partial_sum;
    }
    
    for (i = 0; i < actual_threads; i++) {
        sum += partial_sums[i];
    }
    
    free(partial_sums);
    return step * sum;
}

// Sequential version for baseline comparison
double pi_sequential(long num_steps) {
    double sum = 0.0;
    double x;
    long i;
    
    step = 1.0 / (double) num_steps;
    
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    
    return step * sum;
}

int main(int argc, char **argv) {
    long num_steps = 100000000; // 100 million steps for measurable times
    int thread_counts[] = {1, 2, 4, 8};
    int num_thread_configs = 4;
    double pi, time_start, time_end;
    double seq_time;
    
    // Parse command-line argument for num_steps
    if (argc > 1) {
        num_steps = atol(argv[1]);
        if (num_steps <= 0) {
            fprintf(stderr, "Error: num_steps must be positive\n");
            return 1;
        }
    }
    
    printf("=================================================================\n");
    printf("            Pi Computation Performance Comparison\n");
    printf("=================================================================\n");
    printf("Number of steps: %ld\n\n", num_steps);
    
    // Run sequential version for baseline
    printf("Running sequential version for baseline...\n");
    time_start = omp_get_wtime();
    pi = pi_sequential(num_steps);
    seq_time = omp_get_wtime() - time_start;
    printf("Sequential: Pi = %.10f, Time = %.6f seconds\n\n", pi, seq_time);
    
    // Print table header
    printf("%-20s | %-10s | %-15s | %-10s\n", "Variant", "Threads", "Time (s)", "Speedup");
    printf("--------------------------------------------------------------\n");
    printf("%-20s | %-10s | %-15.6f | %-10s\n", "Sequential", "-", seq_time, "1.00x");
    
    // Test each parallel version with different thread counts
    for (int t = 0; t < num_thread_configs; t++) {
        int threads = thread_counts[t];
        
        // Critical section version
        time_start = omp_get_wtime();
        pi = pi_critical(num_steps, threads);
        time_end = omp_get_wtime() - time_start;
        printf("%-20s | %-10d | %-15.6f | %.2fx\n", "Critical", threads, time_end, seq_time/time_end);
        
        // Atomic version
        time_start = omp_get_wtime();
        pi = pi_atomic(num_steps, threads);
        time_end = omp_get_wtime() - time_start;
        printf("%-20s | %-10d | %-15.6f | %.2fx\n", "Atomic", threads, time_end, seq_time/time_end);
        
        // Reduction version
        time_start = omp_get_wtime();
        pi = pi_reduction(num_steps, threads);
        time_end = omp_get_wtime() - time_start;
        printf("%-20s | %-10d | %-15.6f | %.2fx\n", "Reduction", threads, time_end, seq_time/time_end);
        
        // Manual sum version
        time_start = omp_get_wtime();
        pi = pi_manual_sum(num_steps, threads);
        time_end = omp_get_wtime() - time_start;
        printf("%-20s | %-10d | %-15.6f | %.2fx\n", "Manual Sum", threads, time_end, seq_time/time_end);
        
        printf("--------------------------------------------------------------\n");
    }
    
    printf("\nFinal Pi approximation (reduction): %.10f\n", pi);
    printf("Actual Pi:                          3.1415926535\n");
    
    printf("\n=================================================================\n");
    printf("                        Analysis\n");
    printf("=================================================================\n");
    printf("- Critical: Uses mutex lock, high synchronization overhead\n");
    printf("- Atomic:   Hardware-level atomic operations, less overhead\n");
    printf("- Reduction: OpenMP handles reduction efficiently, best scaling\n");
    printf("- Manual Sum: No synchronization during computation, efficient\n");
    printf("=================================================================\n");
    
    return 0;
}
