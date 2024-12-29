#!/bin/bash
#SBATCH -J dft_job            # Job name
#SBATCH -o dft_job.out        # Standard output file
#SBATCH -e dft_job.err        # Standard error file
#SBATCH -N 1                  # Number of nodes
#SBATCH --ntasks-per-node=4   # Number of tasks (GPUs) per node
#SBATCH -p rtx-dev            # Partition (queue)
#SBATCH -t 2:00:00            # Time limit
#SBATCH --exclusive

# Load necessary modules
module load python3/3.9.2     # Ensure the correct Python module is loaded

# Activate the virtual environment
source $SCRATCH/python-envs/test-env/bin/activate

# Check GPU status
nvidia-smi > gpu_status.log

# Set the environment variable for PyTorch CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the Python script
srun python3 ./main.py

