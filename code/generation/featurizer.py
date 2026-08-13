"""
Builds DTIAM predictor input features for generated candidates.

Concatenates a BerMol compound embedding (computed live from each candidate's
SMILES) with a precomputed ESM-2 protein embedding (looked up by target id in
the protein_features.pkl produced by code/data_process/extract_feature.py),
in the exact column layout code/utils.py's pack() used to build the AutoGluon
training data: unnamed positional float columns, compound vector first.
"""
from typing import Dict, List

try:
    import torch
except Exception:  # pragma: no cover - optional dependency for tests/docs
    torch = None

try:
    import dill as pickle
except Exception:  # pragma: no cover - optional dependency for tests/docs
    import pickle

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency for tests/docs
    pd = None


def _load_bermol_model(path: str, device: str):
    """
    Load a BerMol pickle, remapping any CUDA-tagged tensor storages to
    `device`. BerMolModel_base.pkl was saved from a CUDA-resident model, so
    a plain pickle.load()/dill.load() on a machine or job without CUDA
    access raises "Attempting to deserialize object on a CUDA device but
    torch.cuda.is_available() is False" as soon as it reaches those
    tensors -- torch's own nested torch.load() call (invoked internally by
    the storage objects' unpickling) has no way to know it should remap
    them without map_location, and a plain pickle.load() has no hook to
    pass that through. Temporarily patching torch.load is the standard
    workaround for exactly this.
    """
    original_torch_load = torch.load
    torch.load = lambda *a, **kw: original_torch_load(*a, map_location=torch.device(device), **kw)
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    finally:
        torch.load = original_torch_load


class DTIAMFeatureBuilder:
    """
    Loads a pretrained BerMol compound encoder and a precomputed ESM-2 protein
    feature table, and turns candidate (smiles, target) rows into the feature
    layout the DTIAM AutoGluon predictors were trained on.
    """

    def __init__(
        self,
        bermol_model_path: str,
        protein_features_path: str,
        device: str = "cpu",
    ) -> None:
        if torch is None:
            raise ImportError("torch is required to run the BerMol compound encoder.")
        if pd is None:
            raise ImportError("pandas is required to build DTIAM features.")

        self.device = device

        self._comp_model = _load_bermol_model(bermol_model_path, device)
        self._comp_model.model.to(device)
        self._comp_model.model.eval()

        with open(protein_features_path, "rb") as handle:
            self._protein_features: Dict[str, "np.ndarray"] = pickle.load(handle)

        self._compound_cache: Dict[str, List[float]] = {}

    def _compound_vector(self, smiles: str) -> List[float]:
        if smiles not in self._compound_cache:
            with torch.no_grad():
                _, pooled = self._comp_model.transform(smiles, self.device)
            self._compound_cache[smiles] = pooled.cpu().detach().numpy().reshape(-1).tolist()
        return self._compound_cache[smiles]

    def _protein_vector(self, target: str) -> List[float]:
        if target not in self._protein_features:
            raise KeyError(
                f"No cached ESM-2 features for target '{target}'. Run "
                "code/data_process/extract_feature.py (or an equivalent ESM-2 "
                "extraction step) first, and check that the target id matches "
                "what was used when the protein features were extracted."
            )
        return list(self._protein_features[target])

    def build_batch(self, smiles_target_df: "pd.DataFrame") -> "pd.DataFrame":
        """
        Build a feature DataFrame for a `{smiles, target}` candidate table,
        preserving its index so callers can join predictions back by row.
        """
        if pd is None:
            raise ImportError("pandas is required to build DTIAM features.")
        if "smiles" not in smiles_target_df.columns or "target" not in smiles_target_df.columns:
            raise ValueError("smiles_target_df must contain 'smiles' and 'target' columns.")

        rows = [
            self._compound_vector(smiles) + self._protein_vector(target)
            for smiles, target in zip(smiles_target_df["smiles"], smiles_target_df["target"])
        ]
        return pd.DataFrame(rows, index=smiles_target_df.index)

    def build(self, smiles_list: List[str], target: str) -> "pd.DataFrame":
        """Convenience wrapper for a single-target batch of SMILES."""
        if pd is None:
            raise ImportError("pandas is required to build DTIAM features.")
        return self.build_batch(pd.DataFrame({"smiles": list(smiles_list), "target": target}))
