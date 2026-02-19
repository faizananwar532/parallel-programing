#!/usr/bin/env bash
#SBATCH --job-name="matmul"
#SBATCH --partition=all
#SBATCH --time=0-00:20:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64

####### Output #######
#SBATCH --output=/home/fd0002007/out/matmul_%N_nodes.out.%j
#SBATCH --error=/home/fd0002007/out/matmul_%N_nodes.err.%j

# Usage:
#   sbatch --nodes=1 submit_matmul.sh
#   sbatch --nodes=2 submit_matmul.sh
#   sbatch --nodes=4 submit_matmul.sh
#   sbatch --nodes=6 submit_matmul.sh
#   sbatch --nodes=8 submit_matmul.sh

module load gcc/14.3.0 2>/dev/null
module load openmpi 2>/dev/null

TOTAL_PROCS=$(($SLURM_NNODES * 64))

echo "============================================"
echo "Nodes: $SLURM_NNODES  |  Total processes: $TOTAL_PROCS"
echo "============================================"

echo ""
echo "=== Strategy 1 (matmul — row-block with Bcast+Scatterv+Gatherv) ==="
for run in 1 2 3; do
    echo "--- Run $run ---"
    mpirun -np $TOTAL_PROCS ./matmul 8000 42 0
done

echo ""
echo "=== Strategy 2 (matmul_v2 — k-split with local init + Reduce) ==="
for run in 1 2 3; do
    echo "--- Run $run ---"
    mpirun -np $TOTAL_PROCS ./matmul_v2 8000 42 0
done

echo ""
echo "============================================"
echo "All runs complete."
echo "============================================"
