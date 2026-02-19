#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

double step;

int main(int argc, char **argv) {
    long i;
    long num_steps = 100000; // Default value
    double x, pi, sum = 0.0;
    double time;
    int num_threads = 4;
    double partial_sums[4] = {0.0}; // Array to store partial sums from each thread
    
    // Parse command-line argument for num_steps
    if (argc > 1) {
        num_steps = atol(argv[1]);
        if (num_steps <= 0) {
            fprintf(stderr, "Error: num_steps must be positive\n");
            return 1;
        }
    }
    
    omp_set_num_threads(num_threads);
    time = omp_get_wtime();
    
    step = 1.0 / (double) num_steps;
    
    int actual_threads = 0;
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        double partial_sum = 0.0;
        
        // Store actual number of threads (only first thread needs to do this)
        if (thread_id == 0) {
            actual_threads = total_threads;
        }
        
        // Block decomposition: each thread processes a contiguous block
        long start = (thread_id * num_steps) / total_threads;
        long end = ((thread_id + 1) * num_steps) / total_threads;
        
        for (i = start; i < end; i++) {
            x = (i + 0.5) * step;
            partial_sum = partial_sum + 4.0 / (1.0 + x * x);
        }
        
        // Store partial sum in array (each thread writes to its own index)
        if (thread_id < 4) {
            partial_sums[thread_id] = partial_sum;
        }
    }

    // Combine all partial sums after parallel region
    for (i = 0; i < actual_threads; i++) {
        sum = sum + partial_sums[i];
    }

    pi = step * sum;
    time = omp_get_wtime() - time;
    
    printf("Approximation of Pi:%.10f\n", pi);
    printf("Time: %f seconds with %d threads\n", time, actual_threads);
    printf("Number of steps: %ld\n", num_steps);
    
    return 0;
}

