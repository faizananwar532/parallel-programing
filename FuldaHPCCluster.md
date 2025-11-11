<pre><div class="text_to_html"># Running Programs on the Fulda HPC Cluster

This guide explains how to compile and run OpenMP programs on the Fulda University HPC cluster using the SLURM batch
system.

---

## 1. Editing and Uploading Code

You can either copy your files to the cluster manually or use Git.

### Option A &ndash; Copy via `scp`

Edit your code locally and upload it to the cluster:

```
scp matmul.c fd-your-number@hpc.informatik.hs-fulda.de:~/mySubdir/
```

### Option B - Clone your Git repository on the cluster

Log in and clone your repository directly:

```
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no fd0002007@hpc.informatik.hs-fulda.de
git clone your repo
```

## 2. Login

Connect to the head node via SSH:

```
ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no fd0002007@hpc.informatik.hs-fulda.de
```

## 3. Compilation Example

Compile your code on the login node, but do not execute it there:

```
gcc --version              # Check compiler version
gcc -fopenmp -O3 -Wall -o matmul matmul.c
```

Do not run compute-intensive programs directly on the login node!
Always submit your jobs to the cluster using SLURM.

## 4. Prepare an Output Directory

Before running jobs, create a directory for output files:

```
mkdir -p ~/out
```

## 5. Adjust Your SLURM Submission Script

Open and edit your slurmSubmit.sh file to match your setup.
Ensure your home directory path is correct (`pwd` can show it).

You can edit the script using:
```
nano slurmSubmit.sh
```

Set the desired number of OpenMP threads, for example:
```
export OMP_NUM_THREADS=8
```

## 6. Submit the Job

Submit your job to the scheduler:

```
sbatch slurmSubmit.sh
```

The output (stdout and stderr) will be written to the ~/out directory.

## 7. Check Job Status

Check the status of your own jobs:

```
squeue -u fd-your-number
```

Or view all running jobs:

```
squeue
```

## 8. Cancel Jobs

Cancel a single job:

```
scancel JOBID
```

Cancel all jobs owned by your account:

```
scancel -u fd-your-number
```
</div></pre>