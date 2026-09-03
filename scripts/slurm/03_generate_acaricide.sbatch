#!/bin/bash
#SBATCH --job-name=dtiam_generate_acaricide
#SBATCH --account=mol_thesis
#SBATCH --output=logs/result_%j.out
#SBATCH --error=logs/error_%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=cpu-cvail01
#
# Stage 3: run the genetic-algorithm generation pipeline against all 53
# curated acaricide targets (data/acaricide/targets.csv), producing a
# per-target ranked ligand list (top_k candidates per target, not one
# blended global list -- see CandidateScorer.score_and_rank).
#
# Wall-clock scales with population_size * n_generations * 53 targets; the
# defaults in generation_config.acaricide_example.json (100 x 30) are a
# reasonable first full sweep. Cut population_size/n_generations for a
# faster first pass, then rerun deeper on whichever targets look promising.
#
# No --gres=gpu:1: DTIAMFeatureBuilder defaults to device="cpu" (no "device"
# key set in the acaricide config) and AutoGluon prediction is CPU-only here
# too, so this job never touches the GPU.

set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p /home/juanjo/thesis/DTIAM/logs

# Activate Conda environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate dtiam

# Move to working directory and run
cd /home/juanjo/thesis/DTIAM
python code/generation_pipeline.py \
    --config scripts/generation_config.acaricide_example.json
