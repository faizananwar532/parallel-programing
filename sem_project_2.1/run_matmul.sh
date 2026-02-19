#!/usr/bin/env bash
####### Mail Notify / Job Name / Comment #######
#SBATCH --job-name="matmul"
####### Partition #######
#SBATCH --partition=all
####### Resources #######
#SBATCH --time=0-01:00:00
####### Node Info #######
#SBATCH --exclusive
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=64
#SBATCH --output=/home/fd0002007/parallel-programming/mpi/matmul.out.%j
#SBATCH --error=/home/fd0002007/parallel-programming/mpi/matmul.err.%j

cd /home/fd0002007/parallel-programming/mpi

MPIRUN=/etc/profiles/per-group/cluster/bin/mpirun

#$MPIRUN -np $SLURM_NTASKS ./matmul 8000 42 0

REPEATS=5

for NODES in 1 2 4 6 8; do
    NP=$((NODES * 64))
    echo "=============================="
    echo "Running with $NODES nodes and $NP MPI processes"
    echo "=============================="
    for RUN in $(seq 1 $REPEATS); do
        echo "--- Run $RUN/$REPEATS ---"
        $MPIRUN -np $NP ./matmul 8000 42 0
    done
done