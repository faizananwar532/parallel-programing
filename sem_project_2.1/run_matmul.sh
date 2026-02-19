#!/bin/bash
#SBATCH --job-name=matmul
#SBATCH --output=slurm_matmul_%j.out
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=64
#SBATCH --time=01:00:00
#SBATCH --partition=medium

module load gcc/14.3.0
module load openmpi

make clean
make

N=8000
SEED=42
RUNS=3

echo "========================================================"
echo " Parallel Matrix Multiplication - Speedup Measurement"
echo " Matrix size: ${N}x${N}, Seed: ${SEED}, Runs per config: ${RUNS}"
echo "========================================================"

TMPFILE=$(mktemp)

for NODES in 1 2 4 6 8; do
    NTASKS=$((NODES * 64))

    echo ""
    echo "========================================================"
    echo " Configuration: ${NODES} node(s), ${NTASKS} MPI ranks"
    echo "========================================================"

    echo ""
    echo "--- Verification (n=4, verbose=1) ---"
    srun --nodes=${NODES} --ntasks=${NTASKS} --ntasks-per-node=64 ./matmul 4 ${SEED} 1
    echo ""

    echo "--- Performance (n=${N}, verbose=0) ---"
    SUM=0
    for RUN in $(seq 1 $RUNS); do
        OUTPUT=$(srun --nodes=${NODES} --ntasks=${NTASKS} --ntasks-per-node=64 ./matmul $N $SEED 0)
        echo "$OUTPUT"
        TIME=$(echo "$OUTPUT" | grep "Execution time" | awk '{print $(NF-1)}')
        echo "  Run ${RUN}: ${TIME} s"
        SUM=$(echo "$SUM + $TIME" | bc)
    done
    AVG=$(echo "scale=2; $SUM / $RUNS" | bc)
    echo ""
    echo "  Average time (${NODES} node(s), ${NTASKS} ranks): ${AVG} s"
    echo "${NODES} ${NTASKS} ${AVG}" >> "$TMPFILE"
done

echo ""
echo "========================================================"
echo " SPEEDUP SUMMARY TABLE"
echo "========================================================"
echo ""

BASELINE=$(head -1 "$TMPFILE" | awk '{print $3}')

printf "%-8s %-12s %-14s %-10s\n" "Nodes" "MPI Ranks" "Avg Time (s)" "Speedup"
printf "%-8s %-12s %-14s %-10s\n" "-----" "---------" "------------" "-------"

while read NODES NTASKS AVG; do
    SPEEDUP=$(echo "scale=2; $BASELINE / $AVG" | bc)
    printf "%-8s %-12s %-14s %-10s\n" "$NODES" "$NTASKS" "$AVG" "${SPEEDUP}x"
done < "$TMPFILE"

echo ""
echo "Baseline: 1 node, 64 ranks, ${BASELINE} s"
echo "========================================================"
echo " All measurements complete."
echo "========================================================"

rm -f "$TMPFILE"
