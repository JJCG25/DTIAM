#!/bin/bash
#SBATCH --job-name=dtiam_benchmark_dta_davis
#SBATCH --account=mol_thesis
#SBATCH --output=logs/result_%j.out
#SBATCH --error=logs/error_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --partition=cpu-cvail01
#
# Diagnostic-only: real 5-fold benchmark of the dta/davis model, using the
# same preset/features as the deployed models/dta_davis predictor (best_quality,
# data/dta/davis/features + data_folds/warm_start), but WITHOUT --ensemble-only,
# so each fold trains on its own train_fold_N and is scored against its own
# held-out test_fold_N. Unlike train_models.sh's --ensemble-only run, this
# actually computes RMSE/MSE/Pearson/Spearman/CI per fold (see
# DTIAMModelTrainer._evaluate_fold in train_dtiam_models.py) and writes them to
# <output>/dta_davis_results.csv.
#
# This does NOT touch models/dta_davis (the one top_candidates.csv was scored
# with) -- it trains 5 separate fold predictors into a scratch output dir so
# the production model is untouched. Up to 1h/fold (5h total) at best_quality;
# expect disk usage comparable to 5x the size of models/dta_davis, since each
# fold saves a full bagged AutoGluon ensemble.
#
# No --gres=gpu:1: TabularPredictor.fit() is never given num_gpus anywhere in
# this codebase, so training is CPU-only regardless (see 02_train_models.sbatch).

set -euo pipefail

mkdir -p /home/juanjo/thesis/DTIAM/logs

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate dtiam

cd /home/juanjo/thesis/DTIAM
python code/train_dtiam_models.py \
    --task dta \
    --dataset davis \
    --output results/benchmark/dta_davis \
    --preset best_quality
