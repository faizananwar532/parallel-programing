#include <omp.h>
#include <stdio.h>

static long num_steps = 100000;
double step;

int main() {
    long i;
    double x, pi, sum = 0.0;
    double time;

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
    
    return 0;
}

