"""
Scores REAL, literature-backed ligands (recovered by generation.seed_loader
from the two raw pesticide reference CSVs) against a trained DTIAM predictor,
as a baseline to compare against GA-generated candidates: "how does the model
score known real actives for these targets" vs. "what it thinks about the
molecules the GA invented."

Usage:
    python code/score_reference_ligands.py --config scripts/score_reference_ligands.example.json
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from generation.featurizer import DTIAMFeatureBuilder
from generation.predictor import DTIAMPredictor
from generation.seed_loader import RealLigandSeedLoader

LOGGER = logging.getLogger("score_reference_ligands")


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_targets(config: Dict) -> List[str]:
    """Explicit config['targets'] wins; otherwise read every target_chembl_id
    from a targets CSV (default: data/acaricide/targets.csv, the same 53
    targets top_candidates.csv was generated against, for a direct,
    apples-to-apples comparison)."""
    if config.get("targets"):
        return config["targets"]

    targets_csv = config.get("targets_csv", "data/acaricide/targets.csv")
    targets_df = pd.read_csv(targets_csv, usecols=["target_chembl_id"])
    return sorted(targets_df["target_chembl_id"].unique().tolist())


def run(config: Dict) -> pd.DataFrame:
    task = config.get("task", "dta")
    targets = resolve_targets(config)
    LOGGER.info("Scoring reference ligands for %d targets", len(targets))

    seed_source = config["seed_source"]
    loader = RealLigandSeedLoader(
        relations_csv=seed_source["relations_csv"],
        properties_csv=seed_source["properties_csv"],
    )
    coverage = loader.coverage(targets)
    n_covered = sum(1 for v in coverage.values() if v > 0)
    LOGGER.info(
        "Real ligands found for %d/%d targets (%d total ligand-target pairs)",
        n_covered, len(targets), sum(coverage.values()),
    )

    rows = []
    for target in targets:
        for smi in loader.seeds_for_target(target, max_per_target=config.get("max_per_target")):
            rows.append({"target": target, "smiles": smi})
    pairs_df = pd.DataFrame(rows)
    if pairs_df.empty:
        raise ValueError(
            "No real ligands found for any configured target -- check seed_source paths "
            "and that 'targets' / targets_csv match target_chembl_id values in the relations CSV."
        )

    feature_builder = DTIAMFeatureBuilder(
        bermol_model_path=config["bermol_model_path"],
        protein_features_path=config["protein_features_path"],
        device=config.get("device", "cpu"),
    )
    predictor = DTIAMPredictor(model_paths=config["dtiam_model_paths"])

    LOGGER.info("Scoring %d (ligand, target) pairs...", len(pairs_df))
    features = feature_builder.build_batch(pairs_df[["smiles", "target"]])
    predictions = predictor.predict_all(features)

    result = pairs_df.join(predictions)
    result = result.sort_values(["target", task], ascending=[True, False]).reset_index(drop=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score real reference ligands against DTIAM predictors, as a baseline for GA-generated candidates"
    )
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    config = load_config(args.config)
    task = config.get("task", "dta")

    result = run(config)

    out_dir = Path(config.get("output_dir", "results/generation/reference_ligands"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "reference_ligands_scored.csv"
    result.to_csv(csv_path, index=False)
    LOGGER.info("Exported %d scored reference ligands to %s", len(result), csv_path)

    summary = result.groupby("target")[task].agg(["count", "mean", "min", "max"])
    summary_path = out_dir / "reference_ligands_summary.csv"
    summary.to_csv(summary_path)
    print("\nResumen por target:")
    print(summary)
    print(f"\n{task} global -- media: {result[task].mean():.4f}, mediana: {result[task].median():.4f}")


if __name__ == "__main__":
    main()
