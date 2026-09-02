from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors


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
        """Lipinski Ro5 filter only -- QED is no longer computed/used (see
        ScoreWeights)."""
        rows: List[Dict] = []
        for _, row in candidates.iterrows():
            smiles = row["smiles"]
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            if not self._lipinski_pass(mol):
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
