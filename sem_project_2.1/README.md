# Parallel Matrix Multiplication — MPI (Task 2.1)

## Overview

This program computes C = A × B for square matrices of size n × n using MPI.

**Parallelization Strategy:** Row-wise block distribution with cache-optimized tiled multiplication.

- Each process initializes its own rows of A and the full B matrix locally (deterministic initialization eliminates the need for broadcasting B).
- The multiplication uses a tiled i,k,j loop order with a block size of 64 for cache efficiency.
- Result rows of C are gathered on rank 0 using `MPI_Gatherv`.

## Compilation & Execution (Fulda HPC Cluster)

```bash
module load gcc/14.3.0
module load openmpi
make
```

This runs: `mpicc -O3 -Wall -o matmul matmul.c -lm`

## Running All Measurements

Submit the single Slurm script:

```bash
sbatch run_matmul.sh
```

This script handles everything:
1. Compiles the program on the cluster node
2. For each configuration (1, 2, 4, 6, 8 nodes × 64 ranks/node):
   - Prints matrices A, B, C with n=4 for correctness verification
   - Runs n=8000 three times for timing
3. Outputs a speedup summary table at the end

Output is saved to `slurm_matmul_<jobid>.out`.

## Program Invocation

```bash
srun -n <num_processes> ./matmul <n> <seed> <verbose>
```

**Parameters:**
- `n` — Size of the square matrices (n × n)
- `seed` — Initial value for the random number generator
- `verbose` — If 1 and n ≤ 10, prints matrices A, B, C; if 0, only checksum and time

## Expected Checksum

For seed=42, n=4: `Checksum: 17.502887`

## Files

| File | Description |
|------|-------------|
| `matmul.c` | MPI parallel matrix multiplication implementation |
| `Makefile` | Build configuration (`mpicc -O3 -Wall`) |
| `run_matmul.sh` | Slurm script: compilation + verification + all speedup measurements |
