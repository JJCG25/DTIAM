#!/bin/bash
# Chains the three DTIAM SLURM stages with --dependency=afterok, so each
# stage only starts once the previous one finishes successfully.
#
# Assumes the repo lives at /home/juanjo/thesis/DTIAM and the 'dtiam' conda
# env is already created there (see each .sbatch file's header). Run this
# from that same repo root, on the login node: bash scripts/slurm/submit_all.sh
# SLURM writes logs/result_%j.out relative to wherever sbatch is invoked from,
# so submitting from anywhere else will misplace the log files.

set -euo pipefail
mkdir -p logs

j1=$(sbatch --parsable scripts/slurm/01_extract_features.sbatch)
echo "stage 1 (extract features):        job $j1"

j2=$(sbatch --parsable --dependency=afterok:$j1 scripts/slurm/02_train_models.sbatch)
echo "stage 2 (train predictors):        job $j2 (after $j1)"

j3=$(sbatch --parsable --dependency=afterok:$j2 scripts/slurm/03_generate_acaricide.sbatch)
echo "stage 3 (generate acaricide ligands): job $j3 (after $j2)"

echo
echo "Track progress with: squeue -u \$USER"
echo "Results will land in: results/generation/acaricide/top_candidates.csv"
