#!/bin/bash
#SBATCH --job-name=dtiam_train
#SBATCH --account=mol_thesis
#SBATCH --output=logs/result_%j.out
#SBATCH --error=logs/error_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --partition=prod
#
# Stage 2: train the DTIAM AutoGluon predictors (dti/yamanishi_08,
# dta/davis, moa/activation) used to score generated candidates. Depends on
# stage 1 having produced compound_features.pkl/protein_features.pkl for
# each dataset.
#
# best_quality preset, --ensemble-only (trains once on 100% of the data,
# saved directly to a top-level loadable path -- see train_models.sh).
# Trim --preset in code/train_models.sh (e.g. to good_quality) if you want
# a faster first pass.
#
# No --gres=gpu:1: TabularPredictor.fit() is never given num_gpus anywhere
# in this codebase, so LightGBM/CatBoost/XGBoost train on CPU regardless --
# reserving the GPU here would just hold it idle for up to 12h.

set -euo pipefail

# Create logs directory if it doesn't exist
mkdir -p /home/juanjo/thesis/DTIAM/logs

# Activate Conda environment
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate dtiam

# Move to working directory and run
cd /home/juanjo/thesis/DTIAM
bash code/train_models.sh
