#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100-sxm2:1         # Request 1 V100 GPU
#SBATCH --time=08:00:00                # Adjust as needed
#SBATCH --job-name=bert-meld-gpu
#SBATCH --mem=16G                      # Adjust memory
#SBATCH --ntasks=1
#SBATCH --output=meld_log/bert_output.log
#SBATCH --error=meld_log/bert_error.log
#SBATCH --mail-type=BEGIN,END,FAIL       # Notifications for job start, end, and failure
#SBATCH --mail-user=liu.xinyan1@northeastern.edu

# Load required modules and activate conda env
source ~/.bashrc
conda activate torch-env

# Move to project directory
cd /home/liu.xinyan1/Emotion-Recognition/Base_BERT

# Run training script
python bert_finetune_meld.py

