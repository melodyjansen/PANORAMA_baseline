#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=cseduIMC037
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=logs/inference-%j.out
#SBATCH --error=logs/inference-%j.err

# Activate your virtual environment
source /scratch/fvewijk/venv/nnunet_env/bin/activate
export nnUNet_raw="nnUNet_raw"
export nnUNet_preprocessed="nnUNet_preprocessed"
export nnUNet_results="nnUNet_results"

# Start training: adjust these values to match your use case
nnUNetv2_predict -i PANORAMA_baseline/test/input/images/copy_this/ -o PANORAMA_baseline/test/output -d 100 -c 3d_fullres -f 0
