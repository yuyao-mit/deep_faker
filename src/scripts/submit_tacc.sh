#!/bin/bash
#SBATCH -J deep_faker_train   # Job name
#SBATCH -N 1                  # Number of nodes
#SBATCH --ntasks-per-node=4   # Number of tasks (GPUs) per node
#SBATCH -p rtx-dev            # Partition (queue)
#SBATCH -t 2:00:00            # Time limit
#SBATCH --exclusive

# Generate timestamp for output and error file names
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="/work2/10214/yu_yao/Research_Projects/Microstructure_Enough/deep_faker/src/training_log/tacc_log"
OUT_FILE="${OUTPUT_DIR}/${TIMESTAMP}_train.out"
ERR_FILE="${OUTPUT_DIR}/${TIMESTAMP}_train.err"

# Redirect standard output and error
exec > $OUT_FILE
exec 2> $ERR_FILE

# Load necessary modules
module load python3/3.9.2    

# Activate the virtual environment
source $WORK/pytorch-env/cuda10-home/bin/activate

# Set the environment variable for PyTorch CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the Python script
srun python3 ../main.py
