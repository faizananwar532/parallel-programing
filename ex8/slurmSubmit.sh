#!/usr/bin/env bash
####### Job Name #######
#SBATCH --job-name="fib_sum_tasks"

####### Partition #######
#SBATCH --partition=all

####### Resources #######
#SBATCH --time=0-00:15:00
#SBATCH --cpus-per-task=16

####### Node Info #######
#SBATCH --nodes=1

####### Output #######
#SBATCH --output=/home/fd0002007/out/fib_sum_sweep.out.%j
#SBATCH --error=/home/fd0002007/out/fib_sum_sweep.err.%j

# Fixed parameters
N=45
N_SUM=50000000

# Sweep parameters
CUTOFFS=(10 15 20 25 30)
THREADS_LIST=(8 16 32 64)
GRAINSIZES=(0 10000 25000 50000)
NUM_TASKS_LIST=(0 16 32 64)

# Change to your project directory on the cluster
cd /home/fd0002007/parallel-programing/ex8 || { echo "cd failed"; exit 1; }

# Build once
make

# ============================================
# Task 1: Fibonacci task cutoff sweep
# ============================================
echo "=============================================="
echo "       Task 1: Fibonacci Cutoff Sweep"
echo "=============================================="
echo "n=$N, varying cutoff and threads"
echo ""
printf "%-10s | %-10s | %-15s\n" "Cutoff" "Threads" "Time (s)"
printf "%-10s | %-10s | %-15s\n" "---" "---" "---"

for THREADS in "${THREADS_LIST[@]}"; do
    for CUTOFF in "${CUTOFFS[@]}"; do
        export OMP_NUM_THREADS=${THREADS}
        result=$(./fib_tasks ${N} ${CUTOFF} ${THREADS} 2>&1 | grep "time")
        echo "$result" | while read -r line; do
            printf "%-10d | %-10d | %s\n" "$CUTOFF" "$THREADS" "$(echo "$line" | awk '{print $NF}' | sed 's/ s//')"
        done
    done
done

echo ""
echo "=============================================="
echo "    Task 2: Sum Taskloop Sweep (grainsize)"
echo "=============================================="
echo "n_sum=$N_SUM, varying grainsize and threads"
echo ""
printf "%-15s | %-10s | %-15s | %-15s\n" "Grainsize" "Threads" "For Time (s)" "Loop Time (s)"
printf "%-15s | %-10s | %-15s | %-15s\n" "---" "---" "---" "---"

for THREADS in "${THREADS_LIST[@]}"; do
    for GRAINSIZE in "${GRAINSIZES[@]}"; do
        export OMP_NUM_THREADS=${THREADS}
        result=$(./sum_taskloop ${N_SUM} ${THREADS} ${GRAINSIZE} 0 2>&1)
        for_time=$(echo "$result" | grep "parallel for" | awk '{print $4}')
        task_time=$(echo "$result" | grep "taskloop" | awk '{print $4}')
        printf "%-15d | %-10d | %-15s | %-15s\n" "$GRAINSIZE" "$THREADS" "$for_time" "$task_time"
    done
done

echo ""
echo "=============================================="
echo "    Task 2: Sum Taskloop Sweep (num_tasks)"
echo "=============================================="
echo "n_sum=$N_SUM, varying num_tasks and threads"
echo ""
printf "%-15s | %-10s | %-15s | %-15s\n" "Num_tasks" "Threads" "For Time (s)" "Loop Time (s)"
printf "%-15s | %-10s | %-15s | %-15s\n" "---" "---" "---" "---"

for THREADS in "${THREADS_LIST[@]}"; do
    for NUM_TASKS in "${NUM_TASKS_LIST[@]}"; do
        export OMP_NUM_THREADS=${THREADS}
        result=$(./sum_taskloop ${N_SUM} ${THREADS} 0 ${NUM_TASKS} 2>&1)
        for_time=$(echo "$result" | grep "parallel for" | awk '{print $4}')
        task_time=$(echo "$result" | grep "taskloop" | awk '{print $4}')
        printf "%-15d | %-10d | %-15s | %-15s\n" "$NUM_TASKS" "$THREADS" "$for_time" "$task_time"
    done
done

echo ""
echo "=============================================="
echo "              Sweep Complete"
echo "=============================================="
