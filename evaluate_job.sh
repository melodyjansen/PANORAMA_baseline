#!/bin/bash
#SBATCH --partition=csedu
#SBATCH --account=cseduIMC037
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=7:00:00
#SBATCH --output=logs/postprocess-%j.out
#SBATCH --error=logs/postprocess-%j.err

# Activate your virtual environment
source /scratch/fvewijk/venv/nnunet_env/bin/activate
export nnUNet_raw="nnUNet_raw"
export nnUNet_preprocessed="nnUNet_preprocessed"
export nnUNet_results="nnUNet_results"

# Start training: adjust these values to match your use case
nnUNetv2_evaluate_folder \
  -ref PANORAMA_baseline/test/eval_input \
  -pred PANORAMA_baseline/test/output \
  -l 1 4 \
  -json_output PANORAMA_baseline/post_output

nnUNetv2_evaluate_folder \
  -djfile nnUNet_raw/Dataset100_MyData/dataset.json \
  -pfile nnUNet_results/Dataset100_MyData/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json \
  -o PANORAMA_baseline/post_output/eval_metrics.json \
  PANORAMA_baseline/test/eval_input \
  PANORAMA_baseline/test/output
