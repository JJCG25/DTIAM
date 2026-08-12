from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED


@dataclass
class ScoreWeights:
    dti: float = 0.35
    dta: float = 0.35
    moa: float = 0.15
    qed: float = 0.15


class CandidateScorer:
    def __init__(self, weights: ScoreWeights = ScoreWeights()) -> None:
        self.weights = weights

    @staticmethod
    def _lipinski_pass(mol: Chem.Mol) -> bool:
        if mol is None:
            return False
        return (
            Descriptors.MolWt(mol) <= 500
            and Crippen.MolLogP(mol) <= 5
            and Descriptors.NumHDonors(mol) <= 5
            and Descriptors.NumHAcceptors(mol) <= 10
        )

    def filter_druglike(self, candidates: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict] = []
        for _, row in candidates.iterrows():
            smiles = row["smiles"]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            if not self._lipinski_pass(mol):
                continue
            enriched = row.to_dict()
            enriched["qed"] = float(QED.qed(mol))
            rows.append(enriched)
        return pd.DataFrame(rows)

    def score_and_rank(
        self,
        candidates: pd.DataFrame,
        prediction_df: pd.DataFrame,
        top_k: int,
    ) -> pd.DataFrame:
        merged = candidates.join(prediction_df, how="left")
        for col in ["dti", "dta", "moa", "qed"]:
            if col not in merged.columns:
                merged[col] = 0.0

        merged["score"] = (
            self.weights.dti * merged["dti"]
            + self.weights.dta * merged["dta"]
            + self.weights.moa * merged["moa"]
            + self.weights.qed * merged["qed"]
        )
        merged = merged.sort_values("score", ascending=False)
        return merged.head(top_k).reset_index(drop=True)


def export_candidates(df: pd.DataFrame, csv_path: str, json_path: str) -> None:
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
