"""
Graph-based genetic algorithm for target-conditioned molecule generation.

Evolves a population of molecules directly against a DTIAM predictor score
for a given target protein, using RDKit graph mutation (atom add/delete/swap)
and single-point fragment crossover. Unlike the latent-space Bayesian
optimizer in optimization.py, this requires no encode()/decode() support from
the generator backend -- it operates purely on molecular graphs and treats
the DTIAM predictor as a black-box fitness function.
"""
import random
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency for tests/docs
    np = None

from .featurizer import DTIAMFeatureBuilder

RDLogger.DisableLog("rdApp.*")

_ALLOWED_ATOMS = ["C", "N", "O", "F", "Cl", "Br", "S"]
_PERIODIC_TABLE = Chem.GetPeriodicTable()


def _mutate_change_atom(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Swap one non-aromatic atom's element, keeping the bond graph fixed."""
    idxs = list(range(mol.GetNumAtoms()))
    random.shuffle(idxs)
    for idx in idxs:
        if mol.GetAtomWithIdx(idx).GetIsAromatic():
            continue
        rw = Chem.RWMol(mol)
        atom = rw.GetAtomWithIdx(idx)
        old_num = atom.GetAtomicNum()
        choices = [s for s in _ALLOWED_ATOMS if _PERIODIC_TABLE.GetAtomicNumber(s) != old_num]
        random.shuffle(choices)
        for sym in choices:
            atom.SetAtomicNum(_PERIODIC_TABLE.GetAtomicNumber(sym))
            try:
                Chem.SanitizeMol(rw)
                return rw.GetMol()
            except Exception:
                atom.SetAtomicNum(old_num)
    return None


def _mutate_delete_atom(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Remove one terminal (degree <= 1) atom."""
    if mol.GetNumAtoms() <= 3:
        return None
    idxs = list(range(mol.GetNumAtoms()))
    random.shuffle(idxs)
    for idx in idxs:
        if mol.GetAtomWithIdx(idx).GetDegree() > 1:
            continue
        rw = Chem.RWMol(mol)
        rw.RemoveAtom(idx)
        try:
            Chem.SanitizeMol(rw)
            if rw.GetNumAtoms() > 0:
                return rw.GetMol()
        except Exception:
            continue
    return None


def _mutate_add_atom(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Attach one new atom to an existing atom that has a free valence slot."""
    idxs = list(range(mol.GetNumAtoms()))
    random.shuffle(idxs)
    for idx in idxs:
        if mol.GetAtomWithIdx(idx).GetTotalNumHs() < 1:
            continue
        rw = Chem.RWMol(mol)
        sym = random.choice(_ALLOWED_ATOMS)
        new_idx = rw.AddAtom(Chem.Atom(sym))
        rw.AddBond(idx, new_idx, Chem.BondType.SINGLE)
        try:
            Chem.SanitizeMol(rw)
            return rw.GetMol()
        except Exception:
            continue
    return None


_MUTATIONS = [_mutate_change_atom, _mutate_delete_atom, _mutate_add_atom]


def mutate(mol: Chem.Mol, max_attempts: int = 3) -> Optional[Chem.Mol]:
    """Apply one random graph mutation to `mol`; returns None if none succeed."""
    ops = list(_MUTATIONS)
    random.shuffle(ops)
    for op in ops[:max_attempts]:
        result = op(mol)
        if result is not None:
            return result
    return None


def _acyclic_single_bonds(mol: Chem.Mol) -> List[int]:
    return [
        b.GetIdx() for b in mol.GetBonds()
        if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()
    ]


def crossover(mol_a: Chem.Mol, mol_b: Chem.Mol, max_attempts: int = 10) -> Optional[Chem.Mol]:
    """
    Single-point graph crossover: cut one acyclic single bond in each parent
    and join one resulting fragment from each parent at the cut site.
    """
    bonds_a = _acyclic_single_bonds(mol_a)
    bonds_b = _acyclic_single_bonds(mol_b)
    if not bonds_a or not bonds_b:
        return None

    for _ in range(max_attempts):
        frag_a = Chem.FragmentOnBonds(mol_a, [random.choice(bonds_a)], addDummies=True)
        frag_b = Chem.FragmentOnBonds(mol_b, [random.choice(bonds_b)], addDummies=True)
        pieces_a = Chem.GetMolFrags(frag_a, asMols=True, sanitizeFrags=False)
        pieces_b = Chem.GetMolFrags(frag_b, asMols=True, sanitizeFrags=False)
        if len(pieces_a) != 2 or len(pieces_b) != 2:
            continue

        combo = Chem.RWMol(Chem.CombineMols(random.choice(pieces_a), random.choice(pieces_b)))
        dummy_idxs = [a.GetIdx() for a in combo.GetAtoms() if a.GetAtomicNum() == 0]
        if len(dummy_idxs) != 2:
            continue

        d1, d2 = dummy_idxs
        n1 = combo.GetAtomWithIdx(d1).GetNeighbors()[0].GetIdx()
        n2 = combo.GetAtomWithIdx(d2).GetNeighbors()[0].GetIdx()
        combo.AddBond(n1, n2, Chem.BondType.SINGLE)
        for idx in sorted(dummy_idxs, reverse=True):
            combo.RemoveAtom(idx)

        try:
            mol = combo.GetMol()
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            continue
    return None


class MoleculeGA:
    """
    Graph-based genetic algorithm that evolves a molecule population against a
    black-box fitness function (a DTIAM predictor score for a target protein).
    """

    def __init__(
        self,
        mutation_rate: float = 0.5,
        crossover_rate: float = 0.5,
        elite_fraction: float = 0.1,
        tournament_size: int = 3,
    ) -> None:
        # QED-blended fitness removed on chemist feedback: acaricide ligands
        # need not resemble approved human drugs (QED is calibrated to that
        # profile specifically). Fitness is now the raw predicted task score.
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_fraction = elite_fraction
        self.tournament_size = tournament_size

    def _seed_population(self, seed_smiles: List[str], population_size: int) -> List[Chem.Mol]:
        seed_mols = [m for m in (Chem.MolFromSmiles(s) for s in seed_smiles) if m is not None]
        if not seed_mols:
            raise ValueError("No valid seed SMILES provided to initialize the GA population.")

        population: List[Chem.Mol] = []
        seen = set()
        for mol in seed_mols:
            canon = Chem.MolToSmiles(mol)
            if canon not in seen:
                seen.add(canon)
                population.append(mol)
            if len(population) >= population_size:
                return population

        attempts = 0
        max_attempts = population_size * 20
        while len(population) < population_size and attempts < max_attempts:
            attempts += 1
            child = mutate(random.choice(seed_mols))
            if child is None:
                continue
            canon = Chem.MolToSmiles(child)
            if canon in seen:
                continue
            seen.add(canon)
            population.append(child)

        return population

    def _fitness(
        self,
        population: List[Chem.Mol],
        predictor,
        task: str,
        target: str,
        feature_builder: "DTIAMFeatureBuilder",
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        if np is None:
            raise ImportError("numpy is required to run the genetic algorithm.")

        smiles = [Chem.MolToSmiles(mol) for mol in population]
        features = feature_builder.build(smiles, target)
        task_scores = np.asarray(predictor.predict_all(features)[task].values, dtype=float)
        fitness = task_scores

        return fitness, task_scores

    def _fitness_multi(
        self,
        population: List[Chem.Mol],
        predictor,
        task: str,
        targets: List[str],
        feature_builder: "DTIAMFeatureBuilder",
        aggregation: str = "min",
        target_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple["np.ndarray", "np.ndarray"]:
        """
        Score `population` against every target in `targets` and aggregate
        into a single fitness per molecule -- used for joint/polypharmacology
        optimization instead of one independent GA run per target.

        DTIAMFeatureBuilder caches the BerMol compound embedding per SMILES
        (see featurizer.py), so scoring the same population against N targets
        only repeats the cheap protein-side lookup + predictor call N times,
        not the expensive compound encoding.
        """
        if np is None:
            raise ImportError("numpy is required to run the genetic algorithm.")

        smiles = [Chem.MolToSmiles(mol) for mol in population]

        target_scores = np.zeros((len(population), len(targets)))
        for j, target in enumerate(targets):
            features = feature_builder.build(smiles, target)
            target_scores[:, j] = np.asarray(predictor.predict_all(features)[task].values, dtype=float)

        # Normalize each target's scores to [0, 1] independently before
        # aggregating -- targets can sit on very different score scales, so a
        # raw mean/min would let whichever target has the widest numeric
        # range dominate the aggregate.
        span = target_scores.max(axis=0) - target_scores.min(axis=0) + 1e-8
        normalized = (target_scores - target_scores.min(axis=0)) / span

        if aggregation == "min":
            # Worst-case across targets: rewards candidates that are
            # reasonably good against ALL targets, not just good on average.
            task_component = normalized.min(axis=1)
        elif aggregation == "mean":
            task_component = normalized.mean(axis=1)
        elif aggregation == "weighted":
            if not target_weights:
                raise ValueError("aggregation='weighted' requires target_weights")
            weights = np.array([target_weights[t] for t in targets], dtype=float)
            weights = weights / weights.sum()
            task_component = normalized @ weights
        else:
            raise ValueError(f"Unknown aggregation: {aggregation!r} (expected 'min', 'mean', or 'weighted')")

        fitness = task_component

        return fitness, target_scores

    def _tournament_select(self, population: List[Chem.Mol], fitness: "np.ndarray") -> Chem.Mol:
        contenders = random.sample(range(len(population)), min(self.tournament_size, len(population)))
        winner = max(contenders, key=lambda i: fitness[i])
        return population[winner]

    def run(
        self,
        predictor,
        task: str,
        target: str,
        seed_smiles: List[str],
        feature_builder: "DTIAMFeatureBuilder",
        population_size: int = 100,
        n_generations: int = 30,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Run the GA and return up to `top_k` (SMILES, predicted `task` score)
        pairs from the final population, sorted best-first.
        """
        print(f"\n{'='*70}")
        print(f"Genetic algorithm: optimizing {task} against target {target}")
        print(f"{'='*70}\n")

        population = self._seed_population(seed_smiles, population_size)
        fitness, task_scores = self._fitness(population, predictor, task, target, feature_builder)
        print(f"Generation 0: best {task} = {task_scores.max():.4f} ({len(population)} molecules)")

        for gen in range(1, n_generations + 1):
            n_elite = max(1, int(self.elite_fraction * population_size))
            elite_order = np.argsort(fitness)[::-1][:n_elite]
            next_population = [population[i] for i in elite_order]
            seen = {Chem.MolToSmiles(m) for m in next_population}

            attempts = 0
            max_attempts = population_size * 20
            while len(next_population) < population_size and attempts < max_attempts:
                attempts += 1
                child = None
                if random.random() < self.crossover_rate:
                    parent_a = self._tournament_select(population, fitness)
                    parent_b = self._tournament_select(population, fitness)
                    child = crossover(parent_a, parent_b)

                if child is None:
                    parent = self._tournament_select(population, fitness)
                    child = mutate(parent)
                elif random.random() < self.mutation_rate:
                    mutated = mutate(child)
                    if mutated is not None:
                        child = mutated

                if child is None:
                    continue
                canon = Chem.MolToSmiles(child)
                if canon in seen:
                    continue
                seen.add(canon)
                next_population.append(child)

            population = next_population
            fitness, task_scores = self._fitness(population, predictor, task, target, feature_builder)

            if gen % 5 == 0 or gen == n_generations:
                print(f"Generation {gen}: best {task} = {task_scores.max():.4f}")

        order = np.argsort(fitness)[::-1][:top_k]
        results = [(Chem.MolToSmiles(population[i]), float(task_scores[i])) for i in order]

        print(f"\nGA complete. Top {task} molecule: {results[0][0]} ({results[0][1]:.4f})\n")
        return results

    def run_multi_target(
        self,
        predictor,
        task: str,
        targets: List[str],
        seed_smiles: List[str],
        feature_builder: "DTIAMFeatureBuilder",
        population_size: int = 100,
        n_generations: int = 30,
        top_k: int = 20,
        aggregation: str = "min",
        target_weights: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, Dict[str, float], float]]:
        """
        Run the GA optimizing jointly against multiple target proteins at
        once (polypharmacology-style: one candidate scored against every
        target every generation), instead of one independent run per target
        like `run()`.

        Returns up to `top_k` (smiles, {target: raw_task_score}, fitness)
        tuples from the final population, sorted best-first by the
        aggregated fitness (see `aggregation` in `_fitness_multi`).
        """
        if len(targets) < 2:
            raise ValueError("run_multi_target requires at least 2 targets (use run() for a single target).")

        print(f"\n{'='*70}")
        print(f"Genetic algorithm: multi-target optimization ({aggregation}) against {len(targets)} targets")
        print(f"{'='*70}\n")

        population = self._seed_population(seed_smiles, population_size)
        fitness, target_scores = self._fitness_multi(
            population, predictor, task, targets, feature_builder, aggregation, target_weights
        )
        print(f"Generation 0: best fitness = {fitness.max():.4f} ({len(population)} molecules)")

        for gen in range(1, n_generations + 1):
            n_elite = max(1, int(self.elite_fraction * population_size))
            elite_order = np.argsort(fitness)[::-1][:n_elite]
            next_population = [population[i] for i in elite_order]
            seen = {Chem.MolToSmiles(m) for m in next_population}

            attempts = 0
            max_attempts = population_size * 20
            while len(next_population) < population_size and attempts < max_attempts:
                attempts += 1
                child = None
                if random.random() < self.crossover_rate:
                    parent_a = self._tournament_select(population, fitness)
                    parent_b = self._tournament_select(population, fitness)
                    child = crossover(parent_a, parent_b)

                if child is None:
                    parent = self._tournament_select(population, fitness)
                    child = mutate(parent)
                elif random.random() < self.mutation_rate:
                    mutated = mutate(child)
                    if mutated is not None:
                        child = mutated

                if child is None:
                    continue
                canon = Chem.MolToSmiles(child)
                if canon in seen:
                    continue
                seen.add(canon)
                next_population.append(child)

            population = next_population
            fitness, target_scores = self._fitness_multi(
                population, predictor, task, targets, feature_builder, aggregation, target_weights
            )

            if gen % 5 == 0 or gen == n_generations:
                print(f"Generation {gen}: best fitness = {fitness.max():.4f}")

        order = np.argsort(fitness)[::-1][:top_k]
        results = []
        for i in order:
            smi = Chem.MolToSmiles(population[i])
            per_target = {t: float(target_scores[i, j]) for j, t in enumerate(targets)}
            results.append((smi, per_target, float(fitness[i])))

        print(f"\nGA complete. Top molecule: {results[0][0]} (fitness={results[0][2]:.4f})\n")
        for t, score in results[0][1].items():
            print(f"  {t}: {task}={score:.4f}")
        print()
        return results
