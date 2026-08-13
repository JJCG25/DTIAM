"""
Extract ESM-2 protein features for an arbitrary set of targets -- not tied to
one of the six benchmark datasets in ./data/. Intended for a custom target
list such as the one produced by select_acaricide_targets.py.

Reuses cal_prot_feat from extract_feature.py (same ESM-2 model, same pooling),
so the output protein_features.pkl is in the exact {target_id: vector} format
code/generation/featurizer.py's DTIAMFeatureBuilder expects.

Usage:
    python code/data_process/extract_target_features.py \
        --targets-csv data/acaricide/targets.csv \
        --id-column target_chembl_id \
        --sequence-column secuencia_primaria \
        --output data/acaricide/features/protein_features.pkl
"""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_feature import cal_prot_feat  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ESM-2 features for a custom target list")
    parser.add_argument("--targets-csv", required=True, help="CSV with a target id column and a sequence column")
    parser.add_argument("--id-column", default="target_chembl_id")
    parser.add_argument("--sequence-column", default="secuencia_primaria")
    parser.add_argument("--output", required=True, help="Output protein_features.pkl path")
    args = parser.parse_args()

    targets = pd.read_csv(args.targets_csv)
    targets = targets[[args.id_column, args.sequence_column]].dropna()
    targets.columns = ["pid", "seq"]
    targets = targets.drop_duplicates(subset=["pid"])

    print(f"Extracting ESM-2 features for {len(targets)} targets...")
    prot_feat = cal_prot_feat(targets)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as handle:
        pickle.dump(prot_feat, handle)

    print(f"Saved {len(prot_feat)} target feature vectors -> {out_path}")


if __name__ == "__main__":
    main()
