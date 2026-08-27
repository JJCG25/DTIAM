"""
Multi-target (polypharmacology-style) de novo generation: runs the graph-based
genetic algorithm optimizing a SINGLE population jointly against several
target proteins at once, instead of one independent GA run per target (that's
what generation_pipeline.py's 'genetic_algorithm' strategy does).

Candidate scoring reuses DTIAMFeatureBuilder (BerMol compound embedding +
precomputed ESM-2 protein embedding), exactly like generation_pipeline.py.
Seeding can optionally use real, literature-backed ligands (via
generation.seed_loader.RealLigandSeedLoader) instead of only generic
placeholder molecules -- see scripts/generation_config.multitarget_example.json.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED

from generation.featurizer import DTIAMFeatureBuilder
from generation.genetic_algorithm import MoleculeGA
from generation.predictor import DTIAMPredictor
from generation.seed_loader import RealLigandSeedLoader

LOGGER = logging.getLogger("generation_pipeline_multitarget")


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_seed_smiles(config: Dict) -> List[str]:
    """Real ligand seeds when configured, falling back to config['seed_smiles']
    if none are found (or if use_real_seeds is off)."""
    fallback = config.get("seed_smiles", [])

    if not config.get("use_real_seeds", False):
        return fallback

    seed_source = config.get("seed_source", {})
    loader = RealLigandSeedLoader(
        relations_csv=seed_source["relations_csv"],
        properties_csv=seed_source["properties_csv"],
    )
    LOGGER.info("Real ligand coverage per target: %s", loader.coverage(config["targets"]))

    real_seeds = loader.seeds_for_targets(
        config["targets"], max_per_target=seed_source.get("max_per_target")
    )
    if not real_seeds:
        LOGGER.warning(
            "No real ligand seeds found for targets %s; falling back to generic seed_smiles",
            config["targets"],
        )
        return fallback

    LOGGER.info("Using %d real ligand seeds across %d targets", len(real_seeds), len(config["targets"]))
    return real_seeds


def run_pipeline(config: Dict) -> pd.DataFrame:
    targets = config["targets"]
    if len(targets) < 2:
        raise ValueError(
            "generation_pipeline_multitarget.py requires 2+ targets in 'targets' -- "
            "use generation_pipeline.py for single-target runs."
        )

    task = config.get("task", "dta")
    model_paths = config["dtiam_model_paths"]
    predictor = DTIAMPredictor(model_paths=model_paths)

    feature_builder = DTIAMFeatureBuilder(
        bermol_model_path=config["bermol_model_path"],
        protein_features_path=config["protein_features_path"],
        device=config.get("device", "cpu"),
    )

    seed_smiles = build_seed_smiles(config)
    if not seed_smiles:
        raise ValueError("No seed SMILES available (neither real ligand seeds nor generic seed_smiles).")

    ga_config = config.get("genetic_algorithm", {})
    ga = MoleculeGA(
        mutation_rate=float(ga_config.get("mutation_rate", 0.5)),
        crossover_rate=float(ga_config.get("crossover_rate", 0.5)),
        elite_fraction=float(ga_config.get("elite_fraction", 0.1)),
        tournament_size=int(ga_config.get("tournament_size", 3)),
        qed_weight=float(ga_config.get("qed_weight", 0.3)),
    )

    LOGGER.info("Running multi-target GA against %d targets: %s", len(targets), targets)
    results = ga.run_multi_target(
        predictor=predictor,
        task=task,
        targets=targets,
        seed_smiles=seed_smiles,
        feature_builder=feature_builder,
        population_size=int(ga_config.get("population_size", 100)),
        n_generations=int(ga_config.get("n_generations", 30)),
        top_k=int(ga_config.get("top_k", 20)),
        aggregation=config.get("aggregation", "min"),
        target_weights=config.get("target_weights"),
    )

    rows = []
    for smi, per_target_scores, fitness in results:
        mol = Chem.MolFromSmiles(smi)
        row = {
            "smiles": smi,
            "fitness": fitness,
            "qed": QED.qed(mol) if mol is not None else None,
        }
        for target, score in per_target_scores.items():
            row[f"{task}_{target}"] = score
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="DTIAM multi-target de novo generation pipeline")
    parser.add_argument("--config", required=True, help="Path to multi-target generation config JSON")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger verbosity",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    config = load_config(args.config)

    ranked = run_pipeline(config)

    out_dir = Path(config.get("output_dir", "results/generation/multitarget"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "top_candidates_multitarget.csv"
    json_path = out_dir / "top_candidates_multitarget.json"
    ranked.to_csv(csv_path, index=False)
    ranked.to_json(json_path, orient="records", indent=2)
    LOGGER.info("Exported %d candidates to %s", len(ranked), out_dir)


if __name__ == "__main__":
    main()
