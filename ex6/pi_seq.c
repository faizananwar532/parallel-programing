#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

double step;

int main(int argc, char **argv) {
    long i;
    long num_steps = 100000; // Default value
    double x, pi, sum = 0.0;
    double time;

    // Parse command-line argument for num_steps
    if (argc > 1) {
        num_steps = atol(argv[1]);
        if (num_steps <= 0) {
            fprintf(stderr, "Error: num_steps must be positive\n");
            return 1;
        }
    }

    time = omp_get_wtime();
    
    step = 1.0 / (double) num_steps;
    for (i = 0; i < num_steps; i++) {
        x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }

    pi = step * sum;
    time = omp_get_wtime() - time;
    
    printf("Approximation of Pi:%.10f\n", pi);
    printf("Time: %f seconds\n", time);
    printf("Number of steps: %ld\n", num_steps);
    
    return 0;
}

