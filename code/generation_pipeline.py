"""
End-to-end de novo generation pipeline, with optional target-conditioned
generation for protein-conditioned ligands via either latent-space Bayesian
optimization (requires an encode/decode-capable generator backend) or a
graph-based genetic algorithm (works with any black-box predictor, no
generator backend required).

Candidate scoring runs through DTIAMFeatureBuilder, which reproduces the exact
feature layout the DTIAM AutoGluon predictors were trained on: a BerMol
compound embedding (computed live from each candidate SMILES) concatenated
with a precomputed ESM-2 protein embedding for the target.
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from generation.featurizer import DTIAMFeatureBuilder
from generation.generator import VAEGenerator, load_generation_config
from generation.genetic_algorithm import MoleculeGA
from generation.optimization import LatentSpaceOptimizer
from generation.predictor import DTIAMPredictor
from generation.scorer import CandidateScorer, export_candidates

LOGGER = logging.getLogger("generation_pipeline")


def _build_candidates(smiles: List[str], targets: List[str]) -> pd.DataFrame:
    rows = []
    for target in targets:
        for smi in smiles:
            rows.append({"target": target, "smiles": smi})
    return pd.DataFrame(rows)


def _build_feature_builder(config: Dict) -> Optional[DTIAMFeatureBuilder]:
    bermol_model_path = config.get("bermol_model_path")
    protein_features_path = config.get("protein_features_path")
    if not bermol_model_path or not protein_features_path:
        return None
    return DTIAMFeatureBuilder(
        bermol_model_path=bermol_model_path,
        protein_features_path=protein_features_path,
        device=config.get("device", "cpu"),
    )


def _run_conditional_generation(
    generator: VAEGenerator,
    predictor: DTIAMPredictor,
    targets: List[str],
    opt_config: Dict,
    feature_builder: DTIAMFeatureBuilder,
) -> pd.DataFrame:
    task = opt_config.get("task", "dta")
    optimizer = LatentSpaceOptimizer(latent_dim=int(opt_config.get("latent_dim", 56)))

    frames = []
    for target in targets:
        LOGGER.info("Optimizing latent space for target %s (task=%s)", target, task)
        optimized_smiles = optimizer.optimize_for_property(
            generator=generator,
            predictor=predictor,
            task=task,
            target=target,
            feature_builder=feature_builder,
            n_initial=int(opt_config.get("n_initial", 20)),
            n_iterations=int(opt_config.get("n_iterations", 30)),
            batch_size=int(opt_config.get("batch_size", 10)),
        )
        frames.append(_build_candidates(optimized_smiles, [target]))
    return pd.concat(frames, ignore_index=True)


def _run_genetic_algorithm(
    predictor: DTIAMPredictor,
    targets: List[str],
    ga_config: Dict,
    seed_smiles: List[str],
    feature_builder: DTIAMFeatureBuilder,
) -> pd.DataFrame:
    task = ga_config.get("task", "dta")
    ga = MoleculeGA(
        mutation_rate=float(ga_config.get("mutation_rate", 0.5)),
        crossover_rate=float(ga_config.get("crossover_rate", 0.5)),
        elite_fraction=float(ga_config.get("elite_fraction", 0.1)),
        tournament_size=int(ga_config.get("tournament_size", 3)),
    )

    frames = []
    for target in targets:
        LOGGER.info("Running genetic algorithm for target %s (task=%s)", target, task)
        results = ga.run(
            predictor=predictor,
            task=task,
            target=target,
            seed_smiles=seed_smiles,
            feature_builder=feature_builder,
            population_size=int(ga_config.get("population_size", 100)),
            n_generations=int(ga_config.get("n_generations", 30)),
            top_k=int(ga_config.get("top_k_per_target", 20)),
        )
        smiles = [smi for smi, _ in results]
        frames.append(_build_candidates(smiles, [target]))
    return pd.concat(frames, ignore_index=True)


def run_pipeline(config: Dict) -> pd.DataFrame:
    generator = VAEGenerator.from_dict(config["generator"])
    scorer = CandidateScorer()

    strategy = config.get("strategy", "sampling")
    targets = config.get("targets", [])
    top_k = int(config.get("top_k", 50))
    model_paths = config.get("dtiam_model_paths")

    predictor = DTIAMPredictor(model_paths=model_paths) if model_paths else None

    feature_builder = None
    if predictor is not None:
        feature_builder = _build_feature_builder(config)
        if feature_builder is None:
            raise ValueError(
                "dtiam_model_paths is configured, so bermol_model_path and "
                "protein_features_path are also required to build matching "
                "features for prediction."
            )

    if strategy == "bayesian_opt":
        if predictor is None:
            raise ValueError(
                "strategy 'bayesian_opt' requires dtiam_model_paths with a model "
                "for the optimized task"
            )
        if not targets:
            raise ValueError("strategy 'bayesian_opt' requires at least one target protein")

        candidates = _run_conditional_generation(
            generator=generator,
            predictor=predictor,
            targets=targets,
            opt_config=config.get("conditional_generation", {}),
            feature_builder=feature_builder,
        )
    elif strategy == "genetic_algorithm":
        if predictor is None:
            raise ValueError(
                "strategy 'genetic_algorithm' requires dtiam_model_paths with a model "
                "for the optimized task"
            )
        if not targets:
            raise ValueError("strategy 'genetic_algorithm' requires at least one target protein")
        seed_smiles = config.get("seed_smiles", [])
        if not seed_smiles:
            raise ValueError(
                "strategy 'genetic_algorithm' requires seed_smiles to initialize the population"
            )

        candidates = _run_genetic_algorithm(
            predictor=predictor,
            targets=targets,
            ga_config=config.get("genetic_algorithm", {}),
            seed_smiles=seed_smiles,
            feature_builder=feature_builder,
        )
    else:
        num_samples = int(config.get("num_samples", 256))
        seed_smiles = config.get("seed_smiles", [])
        generation_strategy = config.get("generation_strategy", "sampling")

        LOGGER.info("Generating molecules with %s strategy", generation_strategy)
        generated_smiles = generator.generate(
            num_samples=num_samples,
            strategy=generation_strategy,
            seed_smiles=seed_smiles,
        )
        candidates = _build_candidates(generated_smiles, targets)

    LOGGER.info("Filtering generated molecules by drug-likeness")
    filtered = scorer.filter_druglike(candidates)

    prediction_df = pd.DataFrame(index=filtered.index)
    if predictor is not None:
        LOGGER.info("Running DTIAM batch predictions for tasks: %s", sorted(model_paths))
        features = feature_builder.build_batch(filtered[["smiles", "target"]])
        prediction_df = predictor.predict_all(features)
    else:
        LOGGER.info("No DTIAM models configured; ranking with structural scores only")

    ranked = scorer.score_and_rank(filtered, prediction_df, top_k=top_k)
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="DTIAM de novo generation pipeline")
    parser.add_argument("--config", required=True, help="Path to generation config JSON")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logger verbosity",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    config = load_generation_config(args.config)

    ranked = run_pipeline(config)

    out_dir = Path(config.get("output_dir", "results/generation"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "top_candidates.csv"
    json_path = out_dir / "top_candidates.json"
    export_candidates(ranked, str(csv_path), str(json_path))
    LOGGER.info("Exported %d candidates to %s", len(ranked), out_dir)


if __name__ == "__main__":
    main()
