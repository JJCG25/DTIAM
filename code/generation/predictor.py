from typing import Dict, Optional

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency for tests/docs
    pd = None

try:
    from autogluon.tabular import TabularPredictor
except Exception:  # pragma: no cover - optional dependency for tests/docs
    TabularPredictor = None


class DTIAMPredictor:
    """Load and run pretrained DTIAM AutoGluon predictors."""

    def __init__(self, model_paths: Dict[str, str]) -> None:
        self.model_paths = model_paths
        self._models: Dict[str, object] = {}

    def load(self) -> None:
        if TabularPredictor is None:
            raise ImportError("autogluon is required to load DTIAM predictors.")

        for task, path in self.model_paths.items():
            self._models[task] = TabularPredictor.load(path)

    def _ensure_loaded(self) -> None:
        if not self._models:
            self.load()

    def predict_batch(self, features: "pd.DataFrame", task: str) -> "pd.Series":
        if pd is None:
            raise ImportError("pandas is required to run DTIAM batch predictions.")
        self._ensure_loaded()
        if task not in self._models:
            raise KeyError(f"Task '{task}' model was not loaded.")

        model = self._models[task]
        if task == "dta":
            return model.predict(features)

        probs = model.predict_proba(features)
        return probs.iloc[:, 1]

    def predict_all(self, features: "pd.DataFrame") -> "pd.DataFrame":
        if pd is None:
            raise ImportError("pandas is required to run DTIAM batch predictions.")
        self._ensure_loaded()
        output = pd.DataFrame(index=features.index)
        for task in self._models:
            output[task] = self.predict_batch(features, task)
        return output


def smiles_target_to_features(
    smiles_df: "pd.DataFrame",
    protein_features: Optional["pd.DataFrame"] = None,
) -> "pd.DataFrame":
    """
    Build feature frame expected by DTIAM predictors.

    This utility keeps the interface batch-oriented: pass generated candidate rows
    and merge externally-computed protein features before prediction.
    """
    if pd is None:
        raise ImportError("pandas is required to prepare DTIAM feature tables.")
    if "smiles" not in smiles_df.columns:
        raise ValueError("smiles_df must contain a 'smiles' column.")

    features = smiles_df.copy()
    if protein_features is not None:
        features = features.merge(protein_features, on="target", how="left")
    return features
