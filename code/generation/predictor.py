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
    """Load, run, and (optionally) train pretrained DTIAM AutoGluon predictors."""

    def __init__(self, model_paths: Optional[Dict[str, str]] = None) -> None:
        self.model_paths = model_paths or {}
        self._models: Dict[str, object] = {}

    def load(self) -> None:
        if TabularPredictor is None:
            raise ImportError("autogluon is required to load DTIAM predictors.")

        for task, path in self.model_paths.items():
            self._models[task] = TabularPredictor.load(path)

    def _ensure_loaded(self) -> None:
        if not self._models and self.model_paths:
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

    def train(
        self,
        train_data: "pd.DataFrame",
        task: str,
        eval_metric: Optional[str] = None,
        preset: str = "best_quality",
        time_limit: int = 3600,
    ) -> "TabularPredictor":
        """
        Train a new predictor on provided training data and register it under `task`.

        Args:
            train_data: DataFrame with features and label column 'y'
            task: Task type ('dti', 'dta', 'moa')
            eval_metric: Evaluation metric (e.g., 'roc_auc' for classification)
            preset: AutoGluon preset for model selection
            time_limit: Time limit for training in seconds

        Returns:
            Trained TabularPredictor
        """
        if TabularPredictor is None:
            raise ImportError("AutoGluon is required for training models")

        print(f"Training {task} predictor on {len(train_data)} samples...")

        predictor = TabularPredictor(
            label="y",
            eval_metric=eval_metric,
        ).fit(
            train_data=train_data,
            presets=preset,
            time_limit=time_limit,
        )

        self._models[task] = predictor
        print(f"Training complete for {task}")

        return predictor

    def save_model(self, task: str, save_path: str) -> None:
        """
        Save a trained model to disk.

        Args:
            task: Task type to save
            save_path: Path to save model directory
        """
        if task not in self._models or self._models[task] is None:
            raise ValueError(f"Model for task '{task}' not trained or loaded")

        self._models[task].save(save_path)
        print(f"Saved {task} model to {save_path}")

    def load_model(self, task: str, load_path: str) -> None:
        """
        Load a pretrained model from disk and register it under `task`.

        Args:
            task: Task type to load
            load_path: Path to model directory
        """
        if TabularPredictor is None:
            raise ImportError("AutoGluon is required to load models")

        self._models[task] = TabularPredictor.load(load_path)
        self.model_paths[task] = load_path
        print(f"Loaded {task} model from {load_path}")
