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
#   sbatch --nodes=1 run_matmul.sh
#   sbatch --nodes=2 run_matmul.sh
#   sbatch --nodes=4 run_matmul.sh
#   sbatch --nodes=6 run_matmul.sh
#   sbatch --nodes=8 run_matmul.sh

module load gcc/14.3.0 2>/dev/null
module load openmpi 2>/dev/null

make clean
make

TOTAL_PROCS=$(($SLURM_NNODES * 64))
N=8000
SEED=42
RUNS=3

echo "============================================"
echo "Nodes: $SLURM_NNODES  |  Total processes: $TOTAL_PROCS"
echo "Matrix size: ${N}x${N}, Seed: ${SEED}, Runs: ${RUNS}"
echo "============================================"

echo ""
echo "--- Verification (n=4, verbose=1) ---"
mpirun -np $TOTAL_PROCS ./matmul 4 ${SEED} 1

echo ""
echo "--- Performance (n=${N}, verbose=0) ---"
for run in $(seq 1 $RUNS); do
    echo "--- Run $run ---"
    mpirun -np $TOTAL_PROCS ./matmul $N $SEED 0
done

echo ""
echo "============================================"
echo "All runs complete."
echo "============================================"
