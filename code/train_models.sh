#!/bin/bash
set -euo pipefail

# Train DTIAM models for all tasks and datasets.
#
# Uses --ensemble-only: trains a single predictor on 100% of the data,
# saved directly to --output (loadable via TabularPredictor.load(path)),
# which is what the generation pipeline's dtiam_model_paths config expects.
# Without this flag, train_dtiam_models.py's default is k-fold
# cross-validation, which saves each fold's predictor to output/fold_N/
# instead -- useful for benchmark evaluation, but not directly loadable at
# the top-level path.

echo "Training DTI models..."
python code/train_dtiam_models.py \
    --task dti \
    --dataset yamanishi_08 \
    --output models/dti_yamanishi_08 \
    --preset best_quality \
    --ensemble-only

echo ""
echo "Training DTA models..."
python code/train_dtiam_models.py \
    --task dta \
    --dataset davis \
    --output models/dta_davis \
    --preset best_quality \
    --ensemble-only

echo ""
echo "Training MOA models..."
python code/train_dtiam_models.py \
    --task moa \
    --dataset activation \
    --output models/moa_activation \
    --preset best_quality \
    --ensemble-only

echo ""
echo "All models trained! Check models/ directory for saved predictors."
