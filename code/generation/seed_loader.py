"""
Loads real, literature-backed ligand seeds for the genetic algorithm, instead
of relying only on generic placeholder molecules (ethanol, benzene,
triethylamine, aspirin).

Joins two raw, git-ignored reference files (never committed to this repo --
see .gitignore -- so both paths must be supplied explicitly):

- A ligand-target bioactivity relations export (real compounds tested
  against real targets, with measured affinity and literature DOIs). This
  gives which ligand ChEMBL IDs were tested against a given
  target_chembl_id, but has NO compound structure column at all.
- A unified compound properties table, which has canonical_smiles per
  ChEMBL ID (among many physicochemical/toxicological fields we don't need
  here).

Neither file is required for the rest of the pipeline to work: if a target
has no matching real ligand (absent from either file, or no parseable
SMILES after the join), callers should fall back to their config's generic
seed_smiles.
"""
import logging
from typing import Dict, Iterable, List, Optional

import pandas as pd
from rdkit import Chem

LOGGER = logging.getLogger(__name__)


class RealLigandSeedLoader:
    def __init__(self, relations_csv: str, properties_csv: str) -> None:
        relations = pd.read_csv(relations_csv, usecols=["ligando_chembl_id", "target_chembl_id"])
        properties = pd.read_csv(properties_csv, usecols=["chembl_id", "canonical_smiles"])

        properties = properties.dropna(subset=["canonical_smiles"]).copy()
        valid_mask = properties["canonical_smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
        properties = properties[valid_mask]

        merged = relations.merge(
            properties, left_on="ligando_chembl_id", right_on="chembl_id", how="inner"
        )
        self._by_target: Dict[str, List[str]] = (
            merged.groupby("target_chembl_id")["canonical_smiles"]
            .apply(lambda s: sorted(set(s)))
            .to_dict()
        )
        n_unique_smiles = len({smi for smis in self._by_target.values() for smi in smis})
        LOGGER.info(
            "Loaded real ligand seeds for %d targets (%d unique SMILES total)",
            len(self._by_target),
            n_unique_smiles,
        )

    def seeds_for_targets(self, targets: Iterable[str], max_per_target: Optional[int] = None) -> List[str]:
        """Real, literature-backed SMILES tested against any of `targets`."""
        seeds = set()
        for target in targets:
            matches = self._by_target.get(target, [])
            if max_per_target is not None:
                matches = matches[:max_per_target]
            seeds.update(matches)
        return sorted(seeds)

    def seeds_for_target(self, target: str, max_per_target: Optional[int] = None) -> List[str]:
        return self.seeds_for_targets([target], max_per_target=max_per_target)

    def coverage(self, targets: Iterable[str]) -> Dict[str, int]:
        """How many real ligand seeds were found per target -- useful to log
        before a run, since silently falling back to generic seeds for every
        target defeats the point of this loader."""
        return {target: len(self._by_target.get(target, [])) for target in targets}
