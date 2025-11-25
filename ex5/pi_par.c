#include <omp.h>
#include <stdio.h>

static long num_steps = 100000;
double step;

int main() {
    long i;
    double x, pi, sum = 0.0;
    double time;
    int num_threads = 4;
    double partial_sums[4] = {0.0}; // Array to store partial sums from each thread
    
    omp_set_num_threads(num_threads);
    time = omp_get_wtime();
    
    step = 1.0 / (double) num_steps;
    
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int total_threads = omp_get_num_threads();
        double partial_sum = 0.0;
        
        // Block decomposition: each thread processes a contiguous block
        long start = (thread_id * num_steps) / total_threads;
        long end = ((thread_id + 1) * num_steps) / total_threads;
        
        for (i = start; i < end; i++) {
            x = (i + 0.5) * step;
            partial_sum = partial_sum + 4.0 / (1.0 + x * x);
        }
        
        // Store partial sum in array (each thread writes to its own index)
        partial_sums[thread_id] = partial_sum;
    }

    // Combine all partial sums after parallel region
    for (i = 0; i < num_threads; i++) {
        sum = sum + partial_sums[i];
    }

    pi = step * sum;
    time = omp_get_wtime() - time;
    
    printf("Approximation of Pi:%.10f\n", pi);
    printf("Time: %f seconds with %d threads\n", time, num_threads);
    
    return 0;
}

