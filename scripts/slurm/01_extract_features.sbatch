#!/bin/bash
#SBATCH --job-name=dtiam_extract
#SBATCH --output=logs/result_%j.out
#SBATCH --error=logs/error_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=gpu
#
# Stage 1: feature extraction.
# - BerMol (compound) + ESM-2 (protein) features for the six benchmark
#   datasets (needed to train the DTIAM predictors in stage 2).
# - ESM-2 features for the 53 curated acaricide targets (needed to score
#   generated candidates against them in stage 3).
#
# Running on CPU (--device cpu): the pinned torch==1.12.1 build has no
# compiled kernels for this node's GPU (RTX PRO 6000 Blackwell, sm_120 --
# torch 1.12.1 only supports up to sm_70), so "cuda" fails with
# "no kernel image is available for execution on the device" despite
# torch.cuda.is_available() reporting True. CPU is slower but always works;
# --time is set to this partition's 24h max to give ESM-2-650M room to run
# on ~2000-3000 unique sequences without batching.

# Create logs directory if it doesn't exist
mkdir -p /home/juanjo/thesis/DTIAM/logs

# Activate Conda environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate dtiam

# Move to working directory and run
cd /home/juanjo/thesis/DTIAM/code
bash data_prepare.sh cpu

cd /home/juanjo/thesis/DTIAM
python code/data_process/extract_target_features.py \
    --targets-csv data/acaricide/targets.csv \
    --output data/acaricide/features/protein_features.pkl \
    --device cpu
