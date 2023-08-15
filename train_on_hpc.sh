#!/bin/bash

# Change directory to the project root
cd ~/Git/lyrics-aligner/

# Activate the conda environment
conda activate aligner_train

# Load necessary modules
module load cuda/11.4

# Set wandb to offline mode
export WANDB_MODE=offline

# Run the training script
python train.py --epochs 20 --save_steps 5