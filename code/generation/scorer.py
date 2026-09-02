from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from rdkit import Chem


@dataclass
class ScoreWeights:
    # QED removed on chemist feedback: acaricide ligands need not resemble
    # approved human drugs (QED is calibrated to that profile specifically).
    # Rebalanced from the previous dti=0.35/dta=0.35/moa=0.15/qed=0.15,
    # keeping the same dti:dta:moa ratio (7:7:3) without qed's share.
    dti: float = 0.4
    dta: float = 0.4
    moa: float = 0.2


class CandidateScorer:
    def __init__(self, weights: ScoreWeights = ScoreWeights()) -> None:
        self.weights = weights

    def filter_valid(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
        Drops only unparseable SMILES. No drug-likeness gating (Lipinski Ro5
        and QED were both removed on chemist feedback: acaricide ligands
        need not resemble approved human drugs, so neither should gate or
        weight candidate selection here -- see ScoreWeights).

        In practice this rarely drops anything from GA-generated candidates,
        since MoleculeGA's mutate()/crossover() already only ever return
        RDKit-sanitized molecules; it's here as a defensive check for other
        generator backends (e.g. a VAE strategy) that might emit raw,
        unsanitized SMILES.
        """
        rows: List[Dict] = []
        for _, row in candidates.iterrows():
            if Chem.MolFromSmiles(row["smiles"]) is None:
                continue
            rows.append(row.to_dict())
        return pd.DataFrame(rows)

    def score_and_rank(
        self,
        candidates: pd.DataFrame,
        prediction_df: pd.DataFrame,
        top_k: int,
    ) -> pd.DataFrame:
        """
        Rank candidates by composite score. When multiple distinct 'target'
        values are present, top_k applies per target -- otherwise a single
        target's candidates would crowd out every other target's results
        from one global top_k list.
        """
        merged = candidates.join(prediction_df, how="left")
        for col in ["dti", "dta", "moa"]:
            if col not in merged.columns:
                merged[col] = 0.0

        merged["score"] = (
            self.weights.dti * merged["dti"]
            + self.weights.dta * merged["dta"]
            + self.weights.moa * merged["moa"]
        )
        merged = merged.sort_values("score", ascending=False)

        if "target" in merged.columns and merged["target"].nunique() > 1:
            return merged.groupby("target", sort=False, group_keys=False).head(top_k).reset_index(drop=True)

        return merged.head(top_k).reset_index(drop=True)


def export_candidates(df: pd.DataFrame, csv_path: str, json_path: str) -> None:
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
