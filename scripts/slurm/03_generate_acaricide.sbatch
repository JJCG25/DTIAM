#!/bin/bash
#SBATCH --job-name=dtiam_generate_acaricide
#SBATCH --account=mol_thesis
#SBATCH --output=logs/result_%j.out
#SBATCH --error=logs/error_%j.err
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=cpu-cvail01
#
# --time=05:00:00: WRONG PREVIOUSLY -- job 17381 (post-code-changes, 4 cpus)
# ran with the old --time=03:00:00 and got killed by TIMEOUT at 03:05:04,
# never reaching export_candidates(), so it never wrote a fresh
# top_candidates.csv. The 03:00:00 estimate was based on job 15149's
# 2:23:51 runtime, but that ran with 8 cpus, not the 4 this file dropped to
# in the same commit -- cutting cpus-per-task and --time together, from a
# single 8-cpu data point, didn't leave room for the job actually needing
# more wall-clock at fewer cores. Bumped to 5h with real margin above the
# 3:05:04+ (job was still running, not finished) it needed at 4 cpus --
# tighten again once a run actually completes and reports real Elapsed.
#
# --mem=8G: job 15149's .batch step peaked at MaxRSS=6.45G (~24% headroom
# here). Memory here is mostly fixed-size (population=100, fixed-length
# BerMol/ESM-2 embeddings, one loaded AutoGluon predictor), so it shouldn't
# swing much target to target -- but this is one data point, so watch
# error_%j.err for an OOM kill on the next run and bump back up if it hits.
#
# --cpus-per-task=4: no multiprocessing/joblib anywhere in code/generation/ --
# the GA loop (mutation, crossover, selection) is single-threaded RDKit/Python;
# the only multi-threaded work is PyTorch's intra-op threading during the
# batched BerMol forward pass and whatever AutoGluon's models use internally
# at predict() time, both on small (~population_size) batches where returns
# past a few threads are marginal. Bumped down from 8 -- job 17381 (see
# --time above) suggests this trim wasn't free (took noticeably longer than
# 15149's 8-cpu 2:23:51), so if the next 5h run also comes in a lot slower
# than that, consider bumping cpus-per-task back toward 6-8 instead of
# trusting the "cpus aren't the bottleneck" reasoning below at face value.
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
