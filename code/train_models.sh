#!/bin/bash

# Train DTIAM models for all tasks and datasets

echo "Training DTI models..."
python code/train_dtiam_models.py \
    --task dti \
    --dataset yamanishi_08 \
    --output models/dti_yamanishi_08 \
    --preset best_quality

echo ""
echo "Training DTA models..."
python code/train_dtiam_models.py \
    --task dta \
    --dataset davis \
    --output models/dta_davis \
    --preset best_quality

echo ""
echo "Training MOA models..."
python code/train_dtiam_models.py \
    --task moa \
    --dataset activation \
    --output models/moa_activation \
    --preset best_quality

echo ""
echo "All models trained! Check models/ directory for saved predictors."