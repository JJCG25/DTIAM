"""
Latent-space optimization for conditional molecular generation.

Uses Bayesian optimization (Gaussian Process + Expected Improvement) to find
optimal latent codes that decode to molecules with desired properties.

Two main strategies:
1. Property-guided optimization: Maximize a single property
2. Multi-objective optimization: Balance multiple properties
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
import warnings

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
    from scipy.stats import norm
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not installed. Bayesian optimization disabled.")

from .featurizer import DTIAMFeatureBuilder


class PropertyPredictor:
    """Lightweight predictor for properties in latent space."""
    
    def __init__(self, latent_dim: int, property_name: str = "affinity"):
        """
        Initialize property predictor.
        
        Args:
            latent_dim: Dimension of latent space
            property_name: Name of property to predict
        """
        self.latent_dim = latent_dim
        self.property_name = property_name
        self.gp = None
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit Gaussian Process to latent codes and their properties.
        
        Args:
            X: Latent codes (n_samples, latent_dim)
            y: Property values (n_samples,)
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for Bayesian optimization")
        
        self.X_train = X
        self.y_train = y
        
        # Normalize y
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-8
        y_normalized = (y - self.y_mean) / self.y_std
        
        # Create GP with Matern kernel
        kernel = C(1.0, (1e-3, 1e3)) * Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=False,
            n_restarts_optimizer=5
        )
        self.gp.fit(X, y_normalized)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict property values and uncertainties.
        
        Args:
            X: Latent codes (n_samples, latent_dim)
            
        Returns:
            means: Predicted property values (n_samples,)
            stds: Prediction uncertainties (n_samples,)
        """
        if self.gp is None:
            raise ValueError("Fit the predictor first using .fit()")
        
        y_pred, y_std = self.gp.predict(X, return_std=True)
        
        # Denormalize
        y_pred = y_pred * self.y_std + self.y_mean
        y_std = y_std * self.y_std
        
        return y_pred, y_std
    
    def expected_improvement(self, X: np.ndarray, xi: float = 0.0) -> np.ndarray:
        """
        Compute Expected Improvement acquisition function.
        
        Args:
            X: Latent codes (n_samples, latent_dim)
            xi: Exploration-exploitation trade-off (0 = pure exploitation)
            
        Returns:
            EI values (n_samples,)
        """
        if self.gp is None or self.y_train is None:
            raise ValueError("Fit the predictor first")
        
        y_pred, y_std = self.predict(X)
        
        # Current best (in normalized space)
        y_best = (self.y_train.max() - self.y_mean) / self.y_std
        
        # EI = E[max(0, f(x) - f_best - ξ)]
        with np.errstate(divide='warn'):
            improvement = y_pred - y_best - xi
            Z = improvement / (y_std + 1e-8)
            ei = improvement * norm.cdf(Z) + y_std * norm.pdf(Z)
            ei[y_std == 0.0] = 0.0
        
        return ei


class LatentSpaceOptimizer:
    """Optimize generation in VAE latent space for desired properties."""
    
    def __init__(self, latent_dim: int, latent_bounds: Optional[np.ndarray] = None):
        """
        Initialize optimizer.
        
        Args:
            latent_dim: Dimension of latent space
            latent_bounds: Bounds for latent space (default: [-3, 3] per dimension)
        """
        self.latent_dim = latent_dim
        
        if latent_bounds is None:
            self.latent_bounds = np.array([[-3.0, 3.0]] * latent_dim)
        else:
            self.latent_bounds = latent_bounds
        
        self.property_predictors: Dict[str, PropertyPredictor] = {}
    
    def optimize_for_property(self,
                             generator,
                             predictor,
                             task: str,
                             target: str,
                             feature_builder: "DTIAMFeatureBuilder",
                             n_initial: int = 20,
                             n_iterations: int = 50,
                             batch_size: int = 10) -> List[str]:
        """
        Optimize latent codes to maximize a DTIAM-predicted property for a specific
        target protein, using real encode/decode round-trips through the generator's
        latent space (not random re-sampling).

        Args:
            generator: VAE generator exposing .generate()/.encode()/.decode()
                (see VAEGenerator.encode/decode). Raises AttributeError if the
                loaded backend does not support encode/decode.
            predictor: DTIAM predictor with a .predict_all() method
            task: DTIAM task whose prediction column to optimize ('dti', 'dta', 'moa')
            target: Target protein ID to condition generation on
            feature_builder: DTIAMFeatureBuilder used to turn candidate SMILES into
                the BerMol+ESM-2 feature layout the predictor expects
            n_initial: Number of initial seed samples used to fit the surrogate GP
            n_iterations: Number of Bayesian optimization iterations
            batch_size: Number of latent candidates decoded per iteration

        Returns:
            List containing the best SMILES string found
        """
        print(f"\n{'='*70}")
        print(f"Optimizing {task} against target {target}")
        print(f"{'='*70}\n")

        def _predict(smiles: List[str]) -> "np.ndarray":
            features = feature_builder.build(smiles, target)
            predictions = predictor.predict_all(features)
            return predictions[task].values

        # Seed the surrogate model with real encoded latent codes, not
        # unrelated random noise.
        print(f"Step 1: Generating {n_initial} seed candidates...")
        initial_smiles = generator.generate(n_initial, strategy="sampling")
        initial_z = generator.encode(initial_smiles)
        y_values = _predict(initial_smiles)

        prop_predictor = PropertyPredictor(self.latent_dim, task)
        prop_predictor.fit(initial_z, y_values)

        best_idx = int(y_values.argmax())
        best_z = initial_z[best_idx]
        best_value = y_values[best_idx]
        best_smiles = initial_smiles[best_idx]

        print(f"Initial best {task}: {best_value:.4f}")
        print(f"Best molecule: {best_smiles}\n")

        print(f"Step 2: Running {n_iterations} optimization iterations...")
        X_all = initial_z.copy()
        y_all = y_values.copy()
        all_smiles = list(initial_smiles)

        for iteration in range(n_iterations):
            candidate_z = np.random.uniform(
                self.latent_bounds[:, 0],
                self.latent_bounds[:, 1],
                size=(1000, self.latent_dim)
            )

            ei_scores = prop_predictor.expected_improvement(candidate_z, xi=0.01)
            top_indices = np.argsort(ei_scores)[-batch_size:]
            selected_z = candidate_z[top_indices]

            # Decode the actual selected latent codes back to molecules.
            selected_smiles = generator.decode(selected_z)
            n = min(len(selected_smiles), len(selected_z))
            selected_smiles = selected_smiles[:n]
            selected_z = selected_z[:n]
            if n == 0:
                continue

            selected_y = _predict(selected_smiles)

            X_all = np.vstack([X_all, selected_z])
            y_all = np.append(y_all, selected_y)
            all_smiles.extend(selected_smiles)

            prop_predictor.fit(X_all, y_all)

            current_best_idx = int(y_all.argmax())
            current_best_value = y_all[current_best_idx]

            if current_best_value > best_value:
                best_value = current_best_value
                best_z = X_all[current_best_idx]
                best_smiles = all_smiles[current_best_idx]

            if (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}: Best {task} = {best_value:.4f}")

        print(f"\nOptimization complete!")
        print(f"Final best {task}: {best_value:.4f}")
        print(f"Best molecule: {best_smiles}\n")

        return [best_smiles]
    
    def multi_objective_optimization(self,
                                    properties: List[str],
                                    weights: Dict[str, float],
                                    generator,
                                    predictor,
                                    target: str,
                                    feature_builder: "DTIAMFeatureBuilder",
                                    n_samples: int = 200) -> List[Tuple[str, float]]:
        """
        Multi-objective optimization: generate candidates and rank by weighted score.

        Args:
            properties: DTIAM prediction columns to optimize (e.g. ['dti', 'dta'])
            weights: Weight for each property (should sum to ~1.0)
            generator: VAE generator
            predictor: DTIAM predictor
            target: Target protein ID to condition generation on
            feature_builder: DTIAMFeatureBuilder used to turn candidate SMILES into
                the BerMol+ESM-2 feature layout the predictor expects
            n_samples: Number of candidates to generate

        Returns:
            Sorted list of (SMILES, score) tuples
        """
        print(f"\n{'='*70}")
        print(f"Multi-objective optimization")
        print(f"Properties: {properties}")
        print(f"Weights: {weights}")
        print(f"{'='*70}\n")

        # Generate candidates
        print(f"Generating {n_samples} candidates...")
        smiles_list = generator.generate(n_samples, strategy="sampling")

        # Predict properties
        print(f"Predicting properties...")
        features = feature_builder.build(smiles_list, target)
        predictions = predictor.predict_all(features)

        # Normalize and score
        print(f"Scoring candidates...")
        scores = np.zeros(n_samples)
        
        for prop in properties:
            if prop not in predictions.columns:
                print(f"  Warning: {prop} not in predictions")
                continue
            
            prop_values = predictions[prop].values
            normalized = (prop_values - prop_values.min()) / (prop_values.max() - prop_values.min() + 1e-8)
            scores += weights.get(prop, 0.0) * normalized
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        results = [
            (smiles_list[i], scores[i])
            for i in sorted_indices[:50]  # Return top 50
        ]
        
        print(f"\nTop 5 candidates:")
        for i, (smi, score) in enumerate(results[:5]):
            print(f"  {i+1}. {smi} (score: {score:.4f})")
        
        return results
    
    def interpolation_guided_generation(self,
                                       reference_smiles: str,
                                       generator,
                                       predictor,
                                       direction: str = "high_affinity",
                                       n_interpolations: int = 20) -> List[str]:
        """
        Generate molecules by interpolating toward desired properties.
        
        Args:
            reference_smiles: Reference molecule to interpolate from
            generator: VAE generator
            predictor: DTIAM predictor
            direction: Direction to interpolate ('high_affinity', 'drug_like', etc.)
            n_interpolations: Number of interpolation points
            
        Returns:
            List of interpolated SMILES strings
        """
        raise NotImplementedError(
            "interpolation_guided_generation is not yet implemented: it requires a "
            "well-defined direction vector in latent space (e.g. fit from labeled "
            "high/low-property examples), not just an encode/decode round trip. "
            "Use optimize_for_property for target-conditioned generation instead."
        )