from typing import Dict, List

import pandas as pd

from rdkit import Chem


class CandidateScorer:
    """
    Ranks candidates by predicted dta alone.

    Previously a weighted blend of dti/dta/moa/qed -- QED and Lipinski were
    both removed on chemist feedback (acaricide ligands need not resemble
    approved human drugs), and dti/moa were dropped too: dtiam_model_paths
    only ever configures a dta predictor in practice, so those terms were
    always 0.0 and did nothing except rescale the score column -- dead
    weight, not a real multi-task blend. If dti/moa predictors are wired in
    later, revisit this rather than silently re-adding zero-weighted terms.
    """

    def filter_valid(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """
        Drops only unparseable SMILES. No drug-likeness gating (Lipinski Ro5
        and QED were both removed on chemist feedback: acaricide ligands
        need not resemble approved human drugs, so neither should gate or
        weight candidate selection here).

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
        Rank candidates by dta alone. When multiple distinct 'target' values
        are present, top_k applies per target -- otherwise a single target's
        candidates would crowd out every other target's results from one
        global top_k list.
        """
        merged = candidates.join(prediction_df, how="left")
        if "dta" not in merged.columns:
            raise ValueError(
                "score_and_rank requires a 'dta' prediction column -- ranking is dta-only "
                "now (dti/moa/QED/Lipinski are no longer used), so dtiam_model_paths must "
                "configure a 'dta' predictor."
            )

        merged["score"] = merged["dta"]
        merged = merged.sort_values("score", ascending=False)

        if "target" in merged.columns and merged["target"].nunique() > 1:
            return merged.groupby("target", sort=False, group_keys=False).head(top_k).reset_index(drop=True)

        return merged.head(top_k).reset_index(drop=True)


def export_candidates(df: pd.DataFrame, csv_path: str, json_path: str) -> None:
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
