#!/usr/bin/env bash
####### Job Name #######
#SBATCH --job-name="fib_sum_tasks"

####### Partition #######
#SBATCH --partition=all

####### Resources #######
#SBATCH --time=0-00:05:00
#SBATCH --cpus-per-task=16

####### Node Info #######
#SBATCH --nodes=1

####### Output #######
#SBATCH --output=/home/fd0002007/out/fib_tasks.out.%j
#SBATCH --error=/home/fd0002007/out/fib_tasks.err.%j

# Parameters (override by editing here)
N=45
CUTOFF=25
THREADS=8

# Sum taskloop parameters
N_SUM=50000000      # elements (50M ~ 400 MB)
GRAINSIZE=0         # 0 lets OpenMP choose; set >0 to force chunk size
NUM_TASKS=0         # alternative to grainsize; set >0 to request task count

# Change to your project directory on the cluster
cd /home/fd0002007/parallel-programing/ex8 || { echo "cd failed"; exit 1; }

# Build (fast); remove if you prefer to build manually
make

# Run fib tasks
export OMP_NUM_THREADS=${THREADS}
echo "Running fib_tasks with n=${N}, cutoff=${CUTOFF}, threads=${THREADS}".
./fib_tasks ${N} ${CUTOFF} ${THREADS}

echo ""
echo "=============================================="
echo "Running sum_taskloop (taskloop vs parallel for)"
echo "=============================================="
./sum_taskloop ${N_SUM} ${THREADS} ${GRAINSIZE} ${NUM_TASKS}
