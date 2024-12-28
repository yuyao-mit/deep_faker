#!/bin/bash
#SBATCH --job-name=deep_faker_train       # Job name
#SBATCH --output=deep_faker_%j.log        # Standard output and error log
#SBATCH --gres=gpu:volta:2                # Request 2 Volta GPUs
#SBATCH --cpus-per-task=40                # Request 40 CPUs per task
#SBATCH --time=300:00:00                  # Set maximum runtime

# Load necessary modules
source /etc/profile
module load anaconda/2024a

# Run the Python script
python main.py
