import hashlib
import importlib
import json
import os
import pickle
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np


@dataclass
class PretrainedModelConfig:
    name: str
    weights_url: str
    weights_filename: str
    sha256: Optional[str] = None


class ModelWeightManager:
    def __init__(self, cache_dir: str = ".models") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _checksum(self, file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    def ensure_weights(self, config: PretrainedModelConfig) -> Path:
        target_path = self.cache_dir / config.weights_filename
        if target_path.exists() and config.sha256:
            checksum = self._checksum(target_path)
            if checksum != config.sha256:
                target_path.unlink()

        if not target_path.exists():
            if not config.weights_url:
                raise ValueError(
                    f"Model '{config.name}' requires a weights_url when cached weights are unavailable."
                )
            urllib.request.urlretrieve(config.weights_url, target_path)

        if config.sha256:
            checksum = self._checksum(target_path)
            if checksum != config.sha256:
                target_path.unlink(missing_ok=True)
                raise ValueError(
                    f"Checksum verification failed for '{config.name}'. "
                    "Please verify the configured weights URL/checksum."
                )

        return target_path


class VAEGenerator:
    """Pretrained-model molecular generation without random placeholder logic."""

    def __init__(
        self,
        model_type: str,
        model_config: PretrainedModelConfig,
        cache_dir: str = ".models",
        backend: Optional[Any] = None,
        entrypoint: Optional[str] = None,
    ) -> None:
        self.model_type = model_type.lower()
        self.model_config = model_config
        self.manager = ModelWeightManager(cache_dir)
        self.entrypoint = entrypoint
        self._backend = backend

    @staticmethod
    def _load_callable(entrypoint: str) -> Callable[..., Any]:
        module_name, fn_name = entrypoint.split(":", 1)
        module = importlib.import_module(module_name)
        fn = getattr(module, fn_name)
        if not callable(fn):
            raise TypeError(f"Entrypoint '{entrypoint}' is not callable.")
        return fn

    def _load_backend(self) -> Any:
        if self._backend is not None:
            return self._backend

        weights_path = self.manager.ensure_weights(self.model_config)

        if self.entrypoint:
            loader = self._load_callable(self.entrypoint)
            self._backend = loader(str(weights_path))
            return self._backend

        with weights_path.open("rb") as handle:
            self._backend = pickle.load(handle)

        return self._backend

    @staticmethod
    def _to_smiles_list(result: Any) -> List[str]:
        if isinstance(result, list):
            return [s for s in result if isinstance(s, str)]
        if isinstance(result, tuple):
            return [s for s in result if isinstance(s, str)]
        if isinstance(result, dict):
            smiles = result.get("smiles") or result.get("molecules") or []
            return [s for s in smiles if isinstance(s, str)]
        return []

    def generate(
        self,
        num_samples: int,
        strategy: str = "sampling",
        seed_smiles: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        backend = self._load_backend()
        strategy = strategy.lower()

        if strategy == "sampling":
            if hasattr(backend, "generate"):
                result = backend.generate(number_samples=num_samples)
            elif hasattr(backend, "sample"):
                result = backend.sample(num_samples)
            else:
                raise AttributeError(
                    "Loaded backend does not expose 'generate' or 'sample' methods."
                )
            return self._to_smiles_list(result)

        if strategy == "interpolation":
            seeds = list(seed_smiles or [])
            if len(seeds) < 2:
                raise ValueError("Interpolation strategy requires at least two seed SMILES.")
            if not hasattr(backend, "interpolate"):
                raise AttributeError("Loaded backend does not support interpolation.")
            result = backend.interpolate(seeds[0], seeds[1], num_samples=num_samples, **kwargs)
            return self._to_smiles_list(result)

        if strategy == "optimization":
            seeds = list(seed_smiles or [])
            if not seeds:
                raise ValueError("Optimization strategy requires at least one seed SMILES.")
            optimize_fn = getattr(backend, "optimize", None) or getattr(
                backend, "optimise", None
            )
            if optimize_fn is None:
                raise AttributeError("Loaded backend does not support optimization.")
            result = optimize_fn(seeds, num_samples=num_samples, **kwargs)
            return self._to_smiles_list(result)

        raise ValueError(f"Unsupported generation strategy: {strategy}")

    def encode(self, smiles: Iterable[str]) -> "np.ndarray":
        """Encode SMILES strings to latent vectors using the loaded backend."""
        backend = self._load_backend()
        if not hasattr(backend, "encode"):
            raise AttributeError(
                "Loaded backend does not support encoding SMILES to latent vectors."
            )
        result = backend.encode(list(smiles))
        return np.asarray(result)

    def decode(self, latents: "np.ndarray") -> List[str]:
        """Decode latent vectors back to SMILES strings using the loaded backend."""
        backend = self._load_backend()
        if not hasattr(backend, "decode"):
            raise AttributeError(
                "Loaded backend does not support decoding latent vectors to SMILES."
            )
        result = backend.decode(latents)
        return self._to_smiles_list(result)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "VAEGenerator":
        model_cfg = PretrainedModelConfig(
            name=config["name"],
            weights_url=config.get("weights_url", ""),
            weights_filename=config["weights_filename"],
            sha256=config.get("sha256"),
        )
        return cls(
            model_type=config.get("model_type", config["name"]),
            model_config=model_cfg,
            cache_dir=config.get("cache_dir", ".models"),
            entrypoint=config.get("entrypoint"),
        )


def load_generation_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)
