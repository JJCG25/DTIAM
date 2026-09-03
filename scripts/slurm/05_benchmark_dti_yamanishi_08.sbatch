#!/bin/bash
#SBATCH --job-name=dtiam_benchmark_dti_yamanishi_08
#SBATCH --account=mol_thesis
#SBATCH --output=logs/result_%j.out
#SBATCH --error=logs/error_%j.err
#SBATCH --time=11:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --partition=cpu-cvail01
#
# Diagnostic-only: real 10-fold benchmark of the dti/yamanishi_08 model, same
# preset/features as the deployed models/dti_yamanishi_08 predictor
# (best_quality, data/dti/yamanishi_08/features + data_folds/warm_start), but
# WITHOUT --ensemble-only, so each fold trains on its own train_fold_N and is
# scored against its own held-out test_fold_N -- computes AUROC/AUPR per fold
# (see DTIAMModelTrainer._evaluate_fold) and writes them to
# <output>/dti_yamanishi_08_results.csv.
#
# Does NOT touch models/dti_yamanishi_08 -- trains 10 separate fold predictors
# into a scratch output dir. Up to 1h/fold x 10 folds = up to 10h at
# best_quality; --time is set with headroom above that.
#
# No --gres=gpu:1: TabularPredictor.fit() is never given num_gpus anywhere in
# this codebase, so training is CPU-only regardless (see 02_train_models.sbatch).

set -euo pipefail

mkdir -p /home/juanjo/thesis/DTIAM/logs

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate dtiam

cd /home/juanjo/thesis/DTIAM
python code/train_dtiam_models.py \
    --task dti \
    --dataset yamanishi_08 \
    --output results/benchmark/dti_yamanishi_08 \
    --preset best_quality
