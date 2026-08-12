import argparse
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from generation.generator import VAEGenerator, load_generation_config
from generation.predictor import DTIAMPredictor
from generation.scorer import CandidateScorer, export_candidates

LOGGER = logging.getLogger("generation_pipeline")


def _build_candidates(smiles: List[str], targets: List[str]) -> pd.DataFrame:
    rows = []
    for target in targets:
        for smi in smiles:
            rows.append({"target": target, "smiles": smi})
    return pd.DataFrame(rows)


def run_pipeline(config: Dict) -> pd.DataFrame:
    generator = VAEGenerator.from_dict(config["generator"])
    scorer = CandidateScorer()

    num_samples = int(config.get("num_samples", 256))
    strategy = config.get("strategy", "sampling")
    seed_smiles = config.get("seed_smiles", [])
    targets = config.get("targets", [])
    top_k = int(config.get("top_k", 50))

    LOGGER.info("Generating molecules with %s strategy", strategy)
    generated_smiles = generator.generate(
        num_samples=num_samples,
        strategy=strategy,
        seed_smiles=seed_smiles,
    )
    candidates = _build_candidates(generated_smiles, targets)

    LOGGER.info("Filtering generated molecules by drug-likeness")
    filtered = scorer.filter_druglike(candidates)

    prediction_df = pd.DataFrame(index=filtered.index)
    model_paths = config.get("dtiam_model_paths")
    if model_paths:
        LOGGER.info("Running DTIAM batch predictions for tasks: %s", sorted(model_paths))
        predictor = DTIAMPredictor(model_paths=model_paths)
        prediction_df = predictor.predict_all(filtered)
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

    out_dir = Path(config.get("output_dir", "../results/generation"))
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "top_candidates.csv"
    json_path = out_dir / "top_candidates.json"
    export_candidates(ranked, str(csv_path), str(json_path))
    LOGGER.info("Exported %d candidates to %s", len(ranked), out_dir)


if __name__ == "__main__":
    main()
