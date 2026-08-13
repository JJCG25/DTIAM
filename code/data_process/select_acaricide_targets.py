"""
Selects acaricide/insecticide-MOA-relevant protein targets (and their primary
sequences) from a ligand-target relations export such as
pesticide_ligand_target_relations_PROTEINS_WITH_DOI_FINAL.csv, and writes a
deduplicated {target_chembl_id, proteina_target, secuencia_primaria, moa_class}
table ready for ESM-2 feature extraction via extract_target_features.py.

Usage:
    python code/data_process/select_acaricide_targets.py \
        --relations-csv pesticide_ligand_target_relations_PROTEINS_WITH_DOI_FINAL.csv \
        --output data/acaricide/targets.csv
"""
import argparse
from pathlib import Path

import pandas as pd

# Keyword groups covering the classic IRAC mode-of-action target classes found
# in ChEMBL-derived pesticide bioactivity data. Extend if new classes turn up
# in future relation exports.
MOA_KEYWORDS = {
    "acetylcholinesterase": ["acetylcholinesterase"],
    "gaba_receptor": ["gaba"],
    "nicotinic_acetylcholine_receptor": ["nicotinic acetylcholine receptor"],
    "ryanodine_receptor": ["ryanodine receptor"],
    "mitochondrial_complex_i": ["mitochondrial complex i"],
    "ecdysone_receptor": ["ecdysone receptor"],
    "voltage_dependent_calcium_channel": ["calcium channel"],
}


def select_targets(relations_csv: str) -> pd.DataFrame:
    rel = pd.read_csv(relations_csv)
    name_lower = rel["proteina_target"].fillna("").str.lower()

    moa_class = pd.Series([None] * len(rel), index=rel.index, dtype=object)
    for label, keywords in MOA_KEYWORDS.items():
        mask = name_lower.apply(lambda n: any(k in n for k in keywords))
        moa_class[mask & moa_class.isna()] = label

    selected = rel.loc[
        moa_class.notna(), ["target_chembl_id", "proteina_target", "secuencia_primaria"]
    ].copy()
    selected["moa_class"] = moa_class[moa_class.notna()]
    selected = selected.dropna(subset=["target_chembl_id", "secuencia_primaria"])
    selected = selected.drop_duplicates(subset=["target_chembl_id"]).reset_index(drop=True)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select acaricide-relevant targets from a ligand-target relations CSV"
    )
    parser.add_argument("--relations-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selected = select_targets(args.relations_csv)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.output, index=False)

    print(f"Selected {len(selected)} targets across {selected['moa_class'].nunique()} MOA classes -> {args.output}")
    print(selected["moa_class"].value_counts())


if __name__ == "__main__":
    main()
