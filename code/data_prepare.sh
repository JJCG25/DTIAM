#!/bin/bash
# Usage: bash data_prepare.sh [device]
# device defaults to 'cuda'; pass 'cpu' if your GPU's compute capability is
# newer than what the pinned torch build supports (RuntimeError: no kernel
# image is available for execution on the device).
DEVICE=${1:-cuda}

python data_process/data_split_dti.py
python data_process/data_split_dta.py
python data_process/data_split_moa.py
python data_process/extract_feature.py --device "$DEVICE"
