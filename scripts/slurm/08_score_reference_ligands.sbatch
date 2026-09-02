#!/bin/bash
#SBATCH --job-name=dtiam_score_reference_ligands
#SBATCH --account=mol_thesis
#SBATCH --output=logs/result_%j.out
#SBATCH --error=logs/error_%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu-cvail01
#
# Scores REAL, literature-backed ligands (found via generation/seed_loader.py
# by joining the two raw pesticide reference CSVs) against dta_davis, for all
# 53 acaricide targets (data/acaricide/targets.csv) -- the same universe
# top_candidates.csv was generated against. This is a baseline reference: how
# does the model score known real actives, vs. what it thinks about the
# GA-generated candidates.
#
# Small job: only ~98 (ligand, target) pairs (vs. thousands for the GA runs),
# and BerMol embeddings are cached/batched (see featurizer.py), so this is
# fast even on modest resources.
#
# No --gres=gpu:1: same reasoning as the other generation jobs --
# DTIAMFeatureBuilder defaults to device="cpu" and AutoGluon prediction is
# CPU-only regardless.

set -euo pipefail

mkdir -p /home/juanjo/thesis/DTIAM/logs

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate dtiam

cd /home/juanjo/thesis/DTIAM
python code/score_reference_ligands.py \
    --config scripts/score_reference_ligands.example.json
