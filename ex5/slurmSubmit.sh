#!/usr/bin/env bash
####### Mail Notify / Job Name / Comment #######
#SBATCH --job-name="pi_comparison"

####### Partition #######
#SBATCH --partition=all

####### Ressources #######
#SBATCH --time=0-00:10:00

####### Node Info #######
#SBATCH --exclusive
#SBATCH --nodes=1

####### Output #######
#SBATCH --output=/home/fd0002007/out/pi_comparison.out.%j
#SBATCH --error=/home/fd0002007/out/pi_comparison.err.%j

NUM_STEPS=100000000
cd /home/fd0002007/parallel-programing/ex5
chmod +x pi_seq pi_par pi_par_critical

echo "=============================================="
echo "        Task 1: Sequential vs Parallel"
echo "=============================================="
echo ""
echo "Running sequential version with ${NUM_STEPS} steps:"
./pi_seq ${NUM_STEPS}
echo ""
echo "Running parallel version (manual sum) with ${NUM_STEPS} steps:"
export OMP_NUM_THREADS=4
./pi_par ${NUM_STEPS}

echo ""
echo "=============================================="
echo "  Task 2: Performance Comparison (Critical)"
echo "=============================================="
echo ""
./pi_par_critical ${NUM_STEPS}