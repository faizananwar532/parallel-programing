# Examination 2 — MPI Matrix Multiplication

## Authors
- [Name 1] (fdaiXXXX) — matmul.c (Strategy 1)
- [Name 2] (fdaiYYYY) — matmul_v2.c (Strategy 2)

## Compilation

```bash
module load gcc/14.3.0
module load openmpi
make clean && make
```

## Execution

```bash
mpirun -np <P> ./matmul   n seed verbose
mpirun -np <P> ./matmul_v2 n seed verbose
```

| Parameter | Description |
|-----------|-------------|
| `n`       | Matrix size (n × n) |
| `seed`    | RNG seed |
| `verbose` | 1 = print A, B, C (if n ≤ 10); 0 = checksum only |

### Examples

```bash
mpirun -np 4 ./matmul 4 42 1       # verbose
mpirun -np 64 ./matmul 8000 42 0   # performance run
```

## SLURM Submission

Single script for all node counts — override `--nodes` on the command line:

```bash
sbatch --nodes=1 submit_matmul.sh
sbatch --nodes=2 submit_matmul.sh
sbatch --nodes=4 submit_matmul.sh
sbatch --nodes=6 submit_matmul.sh
sbatch --nodes=8 submit_matmul.sh
```

Each invocation runs both strategies 3 times each for averaging.

## Strategy 1 — Row-block with collectives (matmul.c)

- Rank 0 initializes A and B using `fillArray` (from `row_wise_matrix_mult.c`)
- `MPI_Bcast` broadcasts full B to all processes (1 collective call)
- `MPI_Scatterv` distributes rows of A to processes
- Each process computes its rows of C using ikj loop order
- `MPI_Gatherv` collects result at rank 0

## Strategy 2 — Inner-dimension split with Reduce (matmul_v2.c)

- Each process locally initializes only its needed A-columns and B-rows
  (zero-communication setup, since `my_rand` is deterministic per element)
- Each process computes a partial n×n result matrix
- `MPI_Reduce(MPI_SUM)` combines all partials at rank 0

## Files

- `matmul.c` — Strategy 1
- `matmul_v2.c` — Strategy 2
- `Makefile` — Build rules
- `submit_matmul.sh` — SLURM script (use `--nodes=N` to vary)
- `performance_analysis.tex` — LaTeX template for analysis PDF
- `README.md` — This file

## External Sources

- `my_rand()` and `concatenate()` functions taken from `row_wise_matrix_mult.c` (Moodle)
- MPI functions used as taught in Lectures 09–11
