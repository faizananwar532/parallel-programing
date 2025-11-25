#!/usr/bin/env bash
####### Mail Notify / Job Name / Comment #######
#SBATCH --job-name="pi_seq"

####### Partition #######
#SBATCH --partition=all

####### Ressources #######
#SBATCH --time=0-00:05:00

####### Node Info #######
#SBATCH --exclusive
#SBATCH --nodes=1

####### Output #######
#SBATCH --output=/home/fd0002007/out/pi_seq.out.%j
#SBATCH --error=/home/fd0002007/out/pi_seq.err.%j

export OMP_NUM_THREADS=4
NUM_STEPS=100000
cd /home/fd0002007/parallel-programing/ex5
chmod +x pi_seq pi_par
echo "Running sequential version with $(NUM_STEPS) steps:"
./pi_seq $(NUM_STEPS)
echo ""
echo "Running parallel version with $(NUM_STEPS) steps:"
./pi_par $(NUM_STEPS)
