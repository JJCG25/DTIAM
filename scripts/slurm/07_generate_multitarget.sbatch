#!/bin/bash
#SBATCH --job-name=dtiam_generate_multitarget
#SBATCH --account=mol_thesis
#SBATCH --output=logs/result_%j.out
#SBATCH --error=logs/error_%j.err
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --partition=prod
#
# Multi-target (polypharmacology-style) GA run: one population scored jointly
# against the 7 targets in generation_config.multitarget_example.json -- ONE
# TARGET PER MOA CLASS (acetylcholinesterase, nicotinic AChR, GABA, ryanodine,
# mitochondrial complex I, ecdysone, Ca channel), instead of 53 independent
# single-target runs like 03_generate_acaricide.sbatch. This is the hard,
# cross-mechanism version of multi-target (see the config's _comment) -- a
# hit here would need to work across genuinely different pharmacophores, so
# don't be surprised if 'min' aggregation converges on a mediocre-everywhere
# compromise rather than a strong candidate; that's an expected, reportable
# outcome for this experiment, not a bug. Much less total compute than
# 03_generate_acaricide.sbatch regardless -- 30 generations x 7 targets here,
# vs. 30 generations x 53 targets there -- so this ceiling is comfortable
# headroom, not a tight estimate.
#
# Requires the raw pesticide_ligand_target_relations_PROTEINS_WITH_DOI_FINAL.csv
# and pesticide_properties_unified(1).csv at the repo root (see
# code/generation/seed_loader.py) for real ligand seeds -- falls back to the
# 4 generic seed_smiles in the config if either file is missing or has no
# matches for these 5 targets.
#
# No --gres=gpu:1: same reasoning as 03_generate_acaricide.sbatch --
# DTIAMFeatureBuilder defaults to device="cpu" and AutoGluon prediction is
# CPU-only regardless.

set -euo pipefail

mkdir -p /home/juanjo/thesis/DTIAM/logs

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate dtiam

cd /home/juanjo/thesis/DTIAM
python code/generation_pipeline_multitarget.py \
    --config scripts/generation_config.multitarget_example.json
