import json
import tempfile
import unittest
from pathlib import Path

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

from generation.generator import PretrainedModelConfig, VAEGenerator, load_generation_config

try:
    from generation.scorer import CandidateScorer
except Exception:  # pragma: no cover
    CandidateScorer = None


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
    def test_filter_druglike(self):
        df = pd.DataFrame(
            [{"target": "P1", "smiles": "CCO"}, {"target": "P1", "smiles": "invalid"}]
        )
        scorer = CandidateScorer()
        filtered = scorer.filter_druglike(df)
        self.assertEqual(len(filtered), 1)
        self.assertIn("qed", filtered.columns)


if __name__ == "__main__":
    unittest.main()
