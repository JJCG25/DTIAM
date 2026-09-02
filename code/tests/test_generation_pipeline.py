import json
import tempfile
import unittest
from pathlib import Path

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from generation.generator import PretrainedModelConfig, VAEGenerator, load_generation_config
from generation.predictor import DTIAMPredictor

try:
    from generation.scorer import CandidateScorer
except Exception:  # pragma: no cover
    CandidateScorer = None

try:
    from rdkit import Chem
    from generation.genetic_algorithm import MoleculeGA, crossover, mutate
except Exception:  # pragma: no cover
    Chem = None
    MoleculeGA = crossover = mutate = None


class _FakeBackend:
    def generate(self, number_samples):
        return ["CCO"] * number_samples

    def interpolate(self, start, end, num_samples=2, **kwargs):
        return [start, end][:num_samples]

    def optimize(self, seeds, num_samples=2, **kwargs):
        return seeds[:num_samples]


class GeneratorTests(unittest.TestCase):
    def test_generation_strategies(self):
        cfg = PretrainedModelConfig(
            name="fake", weights_url="", weights_filename="unused.pkl", sha256=None
        )
        generator = VAEGenerator("guacamol", cfg, backend=_FakeBackend())

        sampled = generator.generate(3, strategy="sampling")
        self.assertEqual(len(sampled), 3)

        interpolated = generator.generate(
            2, strategy="interpolation", seed_smiles=["CCO", "CCN"]
        )
        self.assertEqual(interpolated, ["CCO", "CCN"])

        optimized = generator.generate(1, strategy="optimization", seed_smiles=["CCO"])
        self.assertEqual(optimized, ["CCO"])

    def test_load_generation_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "cfg.json"
            payload = {"generator": {"name": "x", "weights_filename": "w.pkl"}}
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_generation_config(str(config_path))
            self.assertEqual(loaded["generator"]["name"], "x")


@unittest.skipIf(pd is None or CandidateScorer is None, "scoring dependencies are missing")
class ScorerTests(unittest.TestCase):
    def test_filter_valid_drops_only_unparseable_smiles(self):
        # A molecule that fails Lipinski Ro5 (MW > 500, e.g.) should still
        # survive now -- Lipinski and QED were both removed on chemist
        # feedback (acaricide ligands need not be drug-like). Only
        # unparseable SMILES get dropped.
        big_mw_smiles = "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
        df = pd.DataFrame(
            [
                {"target": "P1", "smiles": "CCO"},
                {"target": "P1", "smiles": "invalid"},
                {"target": "P1", "smiles": big_mw_smiles},
            ]
        )
        scorer = CandidateScorer()
        filtered = scorer.filter_valid(df)
        self.assertEqual(len(filtered), 2)
        self.assertIn(big_mw_smiles, filtered["smiles"].tolist())
        self.assertNotIn("qed", filtered.columns)

    def test_score_and_rank_is_per_target_when_multiple_targets_present(self):
        # target A's candidates all score higher than target B's -- a single
        # global top_k would previously return zero results for target B.
        candidates = pd.DataFrame(
            [{"target": "A", "smiles": "CCO"}, {"target": "A", "smiles": "CCN"}]
            + [{"target": "B", "smiles": "CCC"}, {"target": "B", "smiles": "CCF"}]
        )
        predictions = pd.DataFrame({"dta": [0.9, 0.8, 0.1, 0.05]}, index=candidates.index)
        scorer = CandidateScorer()
        ranked = scorer.score_and_rank(candidates, predictions, top_k=1)
        self.assertEqual(sorted(ranked["target"]), ["A", "B"])

    def test_score_and_rank_is_global_for_a_single_target(self):
        candidates = pd.DataFrame(
            [{"target": "A", "smiles": "CCO"}, {"target": "A", "smiles": "CCN"}]
        )
        predictions = pd.DataFrame({"dta": [0.9, 0.1]}, index=candidates.index)
        scorer = CandidateScorer()
        ranked = scorer.score_and_rank(candidates, predictions, top_k=1)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked.iloc[0]["smiles"], "CCO")


@unittest.skipIf(pd is None or Chem is None, "GA dependencies are missing")
class GeneticAlgorithmTests(unittest.TestCase):
    def test_mutate_and_crossover_produce_valid_molecules(self):
        mol_a = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        mol_b = Chem.MolFromSmiles("CCN(CC)CC")

        for _ in range(20):
            mutated = mutate(mol_a)
            if mutated is not None:
                self.assertIsNotNone(Chem.MolFromSmiles(Chem.MolToSmiles(mutated)))

        child = None
        for _ in range(20):
            child = crossover(mol_a, mol_b)
            if child is not None:
                break
        self.assertIsNotNone(child, "crossover should succeed at least once in 20 tries")
        self.assertIsNotNone(Chem.MolFromSmiles(Chem.MolToSmiles(child)))

    def test_ga_run_improves_population_and_stays_valid(self):
        class FakePredictor(DTIAMPredictor):
            def predict_all(self, features):
                out = pd.DataFrame(index=features.index)
                out["dta"] = [1.0 if "Cl" in s else 0.0 for s in features["smiles"]]
                return out

        class FakeFeatureBuilder:
            """Stands in for DTIAMFeatureBuilder: skips real BerMol/ESM-2 embedding
            and just passes SMILES through, so the FakePredictor above can score
            them without needing those heavy optional dependencies in tests."""

            def build(self, smiles_list, target):
                return pd.DataFrame({"smiles": list(smiles_list), "target": target})

        ga = MoleculeGA(mutation_rate=0.7, crossover_rate=0.5, elite_fraction=0.2)
        results = ga.run(
            predictor=FakePredictor(model_paths={"dta": "unused"}),
            task="dta",
            target="P00533",
            seed_smiles=["CCO", "c1ccccc1", "CCN(CC)CC"],
            feature_builder=FakeFeatureBuilder(),
            population_size=20,
            n_generations=8,
            top_k=3,
        )
        self.assertTrue(len(results) > 0)
        for smi, score in results:
            self.assertIsNotNone(Chem.MolFromSmiles(smi))
        # the fitness signal exclusively rewards chlorinated molecules,
        # so the top result should have found one within 8 generations.
        self.assertIn("Cl", results[0][0])


if __name__ == "__main__":
    unittest.main()
