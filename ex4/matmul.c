#include <omp.h>
#include <stdio.h>

#define N 700
#define THREADS 2

int main(int argc, char **argv)
{
    int i, j, k;
    double sum = 0.0;
    double a[N][N], b[N][N], c[N][N];
    double time;
    /* Init */
    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            a[i][j] = 3.0 * i + j;
            b[i][j] = 5.2 * i + 2.3 * j;
            c[i][j] = 0.0;
        }
    }
    omp_set_num_threads(THREADS);
    time = omp_get_wtime();
    /* Matrixmultiplication with parallal for-loop*/
    #pragma omp parallel for shared(a, b, c) private(i, j, k) schedule(static)
    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            for (k = 0; k < N; ++k)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    for (i = 0; i < N; ++i)
    {
        for (j = 0; j < N; ++j)
        {
            sum += c[i][j];
        }
    }
    printf("Result: %f\n", sum);
    time = omp_get_wtime() - time;
    printf("Time: %f seconds with %d threads\n", time, THREADS);
    

    return 0;
}