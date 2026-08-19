"""
Train and save DTIAM predictors (DTI, DTA, MOA) using AutoGluon.
Models are trained via k-fold cross-validation and saved for inference.

Usage:
    python code/train_dtiam_models.py --task dti --dataset yamanishi_08 --output models/dti_predictor.pkl
    python code/train_dtiam_models.py --task dta --dataset davis --output models/dta_predictor.pkl
    python code/train_dtiam_models.py --task moa --dataset activation --output models/moa_predictor.pkl
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularPredictor
from utils import load_data, rmse, mse, pearson, spearman, ci, roc_auc, pr_auc
from tqdm import tqdm


class DTIAMModelTrainer:
    """Train and manage DTIAM predictors."""
    
    def __init__(self, task: str, dataset: str, data_root: str = "data/"):
        """
        Initialize trainer.

        Args:
            task: 'dti', 'dta', or 'moa'
            dataset: Dataset name (e.g., 'yamanishi_08', 'davis', 'activation')
            data_root: Root path to data directory (relative to the repo root,
                since this script is invoked as `python code/train_dtiam_models.py`)
        """
        assert task in ["dti", "dta", "moa"], f"Invalid task: {task}"
        self.task = task
        self.dataset = dataset
        self.data_root = data_root
        self.models = {}
        self.results = pd.DataFrame()
        
        # Configure task-specific parameters
        if task == "dti":
            assert dataset in ["yamanishi_08", "hetionet"], f"Invalid DTI dataset: {dataset}"
            self.dataset_path = os.path.join(data_root, "dti", dataset)
            self.k_folds = 10
            self.eval_metric = "roc_auc"
            self.result_cols = ["AUROC", "AUPR"]
        elif task == "dta":
            assert dataset in ["davis", "kiba"], f"Invalid DTA dataset: {dataset}"
            self.dataset_path = os.path.join(data_root, "dta", dataset)
            self.k_folds = 5
            self.eval_metric = None
            self.result_cols = ["RMSE", "MSE", "Pearson", "Spearman", "CI"]
        else:  # moa
            assert dataset in ["activation", "inhibition"], f"Invalid MOA dataset: {dataset}"
            self.dataset_path = os.path.join(data_root, "moa", dataset)
            self.k_folds = 5
            self.eval_metric = "roc_auc"
            self.result_cols = ["AUROC", "AUPR"]
        
        # Load features
        self._load_features()
    
    def _load_features(self):
        """Load precomputed compound and protein features."""
        feat_path = os.path.join(self.dataset_path, "features")
        
        comp_feat_file = os.path.join(feat_path, "compound_features.pkl")
        prot_feat_file = os.path.join(feat_path, "protein_features.pkl")
        
        if not os.path.exists(comp_feat_file) or not os.path.exists(prot_feat_file):
            raise FileNotFoundError(
                f"Feature files not found. Run data_process/extract_feature.py first."
            )
        
        self.comp_feat = pickle.load(open(comp_feat_file, "rb"))
        self.prot_feat = pickle.load(open(prot_feat_file, "rb"))
        print(f"✓ Loaded {len(self.comp_feat)} compound features")
        print(f"✓ Loaded {len(self.prot_feat)} protein features")
    
    def train(self, preset: str = "best_quality", excluded_models=None, save_dir: str = "./models/"):
        """
        Train models on all k-folds.
        
        Args:
            preset: AutoGluon preset ('best_quality', 'high_quality', 'good_quality', 'medium_quality', 'fast_inference')
            excluded_models: List of model types to exclude from AutoGluon ensemble
            save_dir: Directory to save trained models
            
        Returns:
            Dictionary with fold models and average metrics
        """
        if excluded_models is None:
            excluded_models = []
        
        os.makedirs(save_dir, exist_ok=True)
        folds_path = os.path.join(self.dataset_path, "data_folds", "warm_start")
        
        print(f"\n{'='*70}")
        print(f"Training {self.task.upper()} models on {self.dataset} dataset")
        print(f"{'='*70}")
        print(f"Folds: {self.k_folds} | Preset: {preset}")
        print(f"Save directory: {save_dir}\n")
        
        results_list = []
        
        for fold_idx in range(self.k_folds):
            print(f"\n--- Fold {fold_idx + 1}/{self.k_folds} ---")
            
            # Load fold data
            train_data, test_data = load_data(
                folds_path, fold_idx, self.comp_feat, self.prot_feat
            )
            
            print(f"Train shape: {train_data.shape} | Test shape: {test_data.shape}")
            
            # Train predictor
            predictor = TabularPredictor(
                label="y",
                eval_metric=self.eval_metric,
                path=os.path.join(save_dir, f"fold_{fold_idx}")
            ).fit(
                train_data=train_data,
                excluded_model_types=excluded_models,
                presets=preset,
                time_limit=3600  # 1 hour per fold
            )
            
            # Evaluate on test set
            test_data_nolab = test_data.drop(columns=["y"])
            fold_results = self._evaluate_fold(predictor, test_data, test_data_nolab, fold_idx)
            results_list.append(fold_results)
            
            # Save model
            model_path = os.path.join(save_dir, f"fold_{fold_idx}_model.pkl")
            predictor.save(os.path.join(save_dir, f"fold_{fold_idx}"))
            print(f"✓ Saved model to {model_path}")
        
        # Aggregate results
        self.results = pd.DataFrame(results_list)
        print(f"\n{'='*70}")
        print("Training Complete - Results Summary")
        print(f"{'='*70}")
        print(self.results)
        print(f"\nMean results across all folds:")
        print(self.results.mean(axis=0))
        
        # Save results
        results_csv = os.path.join(save_dir, f"{self.task}_{self.dataset}_results.csv")
        self.results.to_csv(results_csv, index=False)
        print(f"✓ Saved results to {results_csv}")
        
        return {
            "results": self.results,
            "mean_metrics": self.results.mean(axis=0),
            "save_dir": save_dir
        }
    
    def _evaluate_fold(self, predictor, test_data, test_data_nolab, fold_idx):
        """Evaluate a single fold and return metrics."""
        fold_results = {"fold": fold_idx + 1}
        
        if self.task == "dta":
            # Regression task
            pred_scores = predictor.predict(test_data_nolab)
            G, P = np.array(test_data["y"]), np.array(pred_scores)
            
            fold_results["RMSE"] = rmse(G, P)
            fold_results["MSE"] = mse(G, P)
            fold_results["Pearson"] = pearson(G, P)
            fold_results["Spearman"] = spearman(G, P)
            fold_results["CI"] = ci(G, P)
            
            print(f"RMSE: {fold_results['RMSE']:.4f} | "
                  f"MSE: {fold_results['MSE']:.4f} | "
                  f"Pearson: {fold_results['Pearson']:.4f} | "
                  f"Spearman: {fold_results['Spearman']:.4f} | "
                  f"CI: {fold_results['CI']:.4f}")
        else:
            # Classification task (DTI, MOA)
            pred_probs = predictor.predict_proba(test_data_nolab)
            y_true = np.array(test_data["y"])
            y_pred = np.array(pred_probs.iloc[:, 1])
            
            fold_results["AUROC"] = roc_auc(y_true, y_pred)
            fold_results["AUPR"] = pr_auc(y_true, y_pred)
            
            print(f"AUROC: {fold_results['AUROC']:.4f} | AUPR: {fold_results['AUPR']:.4f}")
        
        return fold_results
    
    def train_single_ensemble(self, preset: str = "best_quality", 
                             excluded_models=None,
                             save_path: str = "./models/ensemble_model"):
        """
        Train a single ensemble model on all data (not k-fold).
        Useful for final production model.
        
        Args:
            preset: AutoGluon preset
            excluded_models: Model types to exclude
            save_path: Path to save the ensemble model
            
        Returns:
            Trained predictor
        """
        if excluded_models is None:
            excluded_models = []
        
        os.makedirs(save_path, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Training single ensemble {self.task.upper()} model on {self.dataset}")
        print(f"{'='*70}\n")
        
        # Load the full dataset once. train_fold_0 and test_fold_0 are a
        # complete, non-overlapping partition of every labeled pair (KFold's
        # train/test split for fold 0), so their union is the entire dataset
        # exactly once -- concatenating all k train folds instead would
        # duplicate every pair (k-1) times, since each pair sits in every
        # train fold except the single one where it's held out as that
        # fold's test set.
        folds_path = os.path.join(self.dataset_path, "data_folds", "warm_start")
        train_data, test_data = load_data(folds_path, 0, self.comp_feat, self.prot_feat)
        combined_data = pd.concat([train_data, test_data], ignore_index=True)
        print(f"Combined training data shape: {combined_data.shape}")
        
        # Train ensemble
        predictor = TabularPredictor(
            label="y",
            eval_metric=self.eval_metric,
            path=save_path
        ).fit(
            train_data=combined_data,
            excluded_model_types=excluded_models,
            presets=preset,
            time_limit=7200  # 2 hours
        )
        
        print(f"✓ Saved ensemble model to {save_path}")
        return predictor


def main():
    parser = argparse.ArgumentParser(description="Train DTIAM predictors")
    parser.add_argument("--task", type=str, required=True, 
                       choices=["dti", "dta", "moa"],
                       help="Task type")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset name")
    parser.add_argument("--data-root", type=str, default="data/",
                       help="Root path to data directory (relative to the repo root)")
    parser.add_argument("--output", type=str, default="./models/",
                       help="Output directory for trained models")
    parser.add_argument("--preset", type=str, default="best_quality",
                       choices=["best_quality", "high_quality", "good_quality", 
                               "medium_quality", "fast_inference"],
                       help="AutoGluon preset")
    parser.add_argument("--exclude-models", type=str, nargs="+", default=[],
                       help="Model types to exclude from ensemble")
    parser.add_argument("--ensemble-only", action="store_true",
                       help="Train single ensemble on all data (no k-fold)")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = DTIAMModelTrainer(args.task, args.dataset, args.data_root)
    
    # Train models
    if args.ensemble_only:
        trainer.train_single_ensemble(
            preset=args.preset,
            excluded_models=args.exclude_models,
            save_path=args.output
        )
    else:
        trainer.train(
            preset=args.preset,
            excluded_models=args.exclude_models,
            save_dir=args.output
        )


if __name__ == "__main__":
    main()