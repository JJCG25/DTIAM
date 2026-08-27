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

        # BerMolPreTrainer.transform() builds a fresh BerMolTokenizer(self.vocab)
        # on every single call (see BerMol/bermol/trainer.py) -- built once
        # here instead, and reused for every molecule/batch.
        from bermol.tokenizer import BerMolTokenizer

        self._tokenizer = BerMolTokenizer(self._comp_model.vocab)
        self._pad_token_id = self._comp_model.vocab.pad_index

        with open(protein_features_path, "rb") as handle:
            self._protein_features: Dict[str, "np.ndarray"] = pickle.load(handle)

        self._compound_cache: Dict[str, List[float]] = {}

    def _compound_vectors_batch(self, smiles_list: List[str]) -> None:
        """
        Populate self._compound_cache for every not-yet-cached SMILES in
        smiles_list with ONE batched BerMol forward pass, instead of one
        forward pass per molecule (BerMolPreTrainer.transform() only accepts
        a single SMILES -- see BerMol/bermol/trainer.py:117). A single tiny
        per-molecule forward pass is too small to use more than a sliver of
        multiple reserved CPUs; one batched matmul over N molecules actually
        does.

        Correctness note: BerMolEncoder pads on the right and the pooled
        output only reads the CLS token at position 0 (BertPooler), which is
        never a padded position -- so padding length doesn't change a given
        molecule's pooled embedding, only which other molecules share its
        batch.
        """
        new_smiles = [s for s in dict.fromkeys(smiles_list) if s not in self._compound_cache]
        if not new_smiles:
            return

        token_seqs, valid_smiles = [], []
        for smi in new_smiles:
            try:
                token_seqs.append(self._tokenizer.encode(smi).squeeze(0))
                valid_smiles.append(smi)
            except Exception:
                # Invalid SMILES: leave uncached. build_batch's cache lookup
                # will then KeyError loudly on it via a plain dict access,
                # rather than silently caching garbage features.
                continue

        if not token_seqs:
            return

        max_len = max(seq.shape[0] for seq in token_seqs)
        padded = torch.full((len(token_seqs), max_len), self._pad_token_id, dtype=torch.long)
        # Additive attention mask: 0.0 for real tokens, large negative for
        # padding, matching BertSelfAttention's `scores + attention_mask`.
        # Shape must be [batch, 1, seq_len]: BertSelfAttention does exactly
        # one torch.unsqueeze(attention_mask, 1) internally, so a plain
        # [batch, seq_len] 2D mask ends up broadcast against
        # [batch, heads, seq_q, seq_k] one dimension short -- it silently
        # aligns the batch axis against the heads axis instead of erroring
        # (or worse, produces wrong-but-plausible numbers whenever
        # batch_size happens to equal num_attention_heads). Verified
        # numerically: this 3D shape reproduces the unpadded single-molecule
        # pooled output to float32 precision; a 2D mask does not.
        attention_mask = torch.zeros((len(token_seqs), 1, max_len), dtype=torch.float)
        for i, seq in enumerate(token_seqs):
            seq_len = seq.shape[0]
            padded[i, :seq_len] = seq
            attention_mask[i, 0, seq_len:] = -10000.0

        padded = padded.to(self.device)
        attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            _, pooled = self._comp_model.model.encoder(padded, attention_mask)

        pooled = pooled.cpu().detach().numpy()
        for i, smi in enumerate(valid_smiles):
            self._compound_cache[smi] = pooled[i].reshape(-1).tolist()

    def _compound_vector(self, smiles: str) -> List[float]:
        if smiles not in self._compound_cache:
            # Fallback for direct single-molecule calls; build_batch below
            # always precomputes the whole batch first, so this is normally
            # already a cache hit by the time it runs.
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

        self._compound_vectors_batch(smiles_target_df["smiles"].tolist())
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
