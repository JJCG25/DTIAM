#!/bin/bash
set -euo pipefail

# Example usage for de novo generation + DTIAM prediction pipeline
python code/generation_pipeline.py --config scripts/generation_config.example.json
