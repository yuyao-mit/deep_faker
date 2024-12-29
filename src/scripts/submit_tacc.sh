#!/bin/bash
#SBATCH -J deep_faker_train   # Job name
#SBATCH -o /work2/10214/yu_yao/Research_Projects/Microstructure_Enough/deep_faker/src/training_log/tacc_log/dft_tacc_$(date +%Y%m%d_%H%M%S).out # Standard output file with timestamp
#SBATCH -e /work2/10214/yu_yao/Research_Projects/Microstructure_Enough/deep_faker/src/training_log/tacc_log/dft_tacc_$(date +%Y%m%d_%H%M%S).err # Standard error file with timestamp
#SBATCH -N 1                  # Number of nodes
#SBATCH --ntasks-per-node=4   # Number of tasks (GPUs) per node
#SBATCH -p rtx-dev            # Partition (queue)
#SBATCH -t 2:00:00            # Time limit
#SBATCH --exclusive

# Load necessary modules
module load python3/3.9.2    

# Activate the virtual environment
source $WORK/pytorch-env/cuda10-home/bin/activate

# Set the environment variable for PyTorch CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the Python script
srun python3 ../main.py