#!/usr/bin/env python3
"""
Nested cross-validation plus final model training (Option 2).

Workflow:
1. Build multi-class labels from the beer consumption dataset.
2. Run nested CV to obtain an unbiased estimate of generalisation accuracy.
3. Re-run a hyperparameter search on the full dataset (outer folds collapsed)
   to confirm the best configuration.
4. Retrain the final pipeline on all available data with the selected hyperparameters.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.base import BaseEstimator

# Ensure immediate console output (important for VS Code / Jupyter)
sys.stdout.reconfigure(line_buffering=True)

# Default tqdm settings for consistent appearance across environments
tqdm_kwargs = dict(file=sys.stdout, ncols=90, dynamic_ncols=True)


# ---------------------------------------------------------------------
# Data class for storing results
# ---------------------------------------------------------------------
@dataclass
class NestedCVReport:
    outer_accuracy: float
    chosen_params: Iterable[dict]


# ---------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------
def load_dataset(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load data, derive multi-class labels, and return features/target."""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df["Class1"] = (df["Litres"] > 29000).astype(int)
    df["Class2"] = (df["Litres"] > 25000).astype(int)
    df["Class3"] = (df["Litres"] > 22000).astype(int)
    df["Class4"] = (df["Litres"] > 10000).astype(int)
    df["Class"] = df[["Class1", "Class2", "Class3", "Class4"]].sum(axis=1)

    # Month cyclic encoding
    df["Month"] = df["Date"].dt.month
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    feature_cols = [
        "AvgTemp",
        "MinTemp",
        "MaxTemp",
        "Rainfall_mm",
        "Weekend",
        "Month_sin",
        "Month_cos",
    ]
    X = df[feature_cols]
    y = df["Class"]
    return X, y


# ---------------------------------------------------------------------
# Pipeline and parameter grid
# ---------------------------------------------------------------------
class PlaceholderModel(BaseEstimator):
    """Dummy model used only as a placeholder in the pipeline."""
    def fit(self, X, y): return self
    def predict(self, X): return np.zeros(len(X))

def make_base_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", PlaceholderModel()),  # explicit placeholder
    ])


def build_param_grid(random_state: int = 42) -> list[dict]:
    """Return a list of grid configurations spanning multiple algorithms."""
    return [
        {
            "scaler": [StandardScaler()],
            "model": [LogisticRegression(max_iter=1000, solver="lbfgs")],
            "model__C": [0.01, 0.1, 1.0, 10.0],
        },
        {
            "scaler": [StandardScaler()],
            "model": [KNeighborsClassifier()],
            "model__n_neighbors": [3, 5, 7, 9],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        {
            "scaler": ["passthrough"],
            "model": [DecisionTreeClassifier(random_state=random_state)],
            "model__max_depth": [None, 3, 5, 7, 9],
            "model__min_samples_leaf": [1, 3, 5, 10],
        },
        {
            "scaler": ["passthrough"],
            "model": [
                RandomForestClassifier(random_state=random_state, n_jobs=-1)
            ],
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [None, 5, 10],
            "model__min_samples_leaf": [1, 3, 5],
        },
    ]


# ---------------------------------------------------------------------
# Nested cross-validation
# ---------------------------------------------------------------------
def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = 42,
) -> NestedCVReport:
    """Execute nested CV with tqdm progress bars and timing."""
    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=random_state + 1)

    base_pipeline = make_base_pipeline()
    param_grid = build_param_grid(random_state)

    outer_scores = []
    chosen_params = []

    tqdm.write("\n=== Nested Cross-Validation (Performance Estimation) ===", file=sys.stdout)
    with tqdm(total=outer_cv.get_n_splits(), desc="Outer folds", **tqdm_kwargs) as pbar:
        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
            t0 = time.time()

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            grid = GridSearchCV(
                estimator=base_pipeline,
                param_grid=param_grid,
                cv=inner_cv,
                scoring="f1_macro",
                n_jobs=-1,
            )

            tqdm.write(f" → Fold {fold_idx}: running inner GridSearchCV...", file=sys.stdout)
            grid.fit(X_train, y_train)

            y_pred = grid.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average="macro")
            best_model = grid.best_estimator_.named_steps["model"]
            best_params = grid.best_params_

            elapsed = time.time() - t0
            tqdm.write(
                f"   Fold {fold_idx} done in {elapsed:.1f}s | "
                f"Model={type(best_model).__name__:<20} | "
                f"Params={best_params} | "
                f"Outer acc={accuracy:.3f} | Outer F1={macro_f1:.3f}",
                file=sys.stdout,
            )
            outer_scores.append(accuracy)
            chosen_params.append(
                {
                    "fold": fold_idx,
                    "best_params": grid.best_params_,
                    "inner_accuracy": grid.best_score_,
                    "outer_accuracy": accuracy,
                    "outer_macro_f1": macro_f1,
                    "selected_model": type(best_model).__name__,
                    "elapsed_sec": elapsed,
                }
            )

            pbar.update(1)

    mean_outer_accuracy = float(np.mean(outer_scores))
    std_outer_accuracy = float(np.std(outer_scores))
    tqdm.write(
        f"\nNested CV complete: Mean outer accuracy = "
        f"{mean_outer_accuracy:.3f} ± {std_outer_accuracy:.3f}",
        file=sys.stdout,
    )

    return NestedCVReport(outer_accuracy=mean_outer_accuracy, chosen_params=chosen_params)


# ---------------------------------------------------------------------
# Full-data hyperparameter search
# ---------------------------------------------------------------------
def hyperparameter_search_full_data(
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits: int = 5,
    random_state: int = 42,
) -> GridSearchCV:
    """Perform a final grid search on the full dataset with a tqdm progress bar."""
    tqdm.write("\n=== Hyperparameter Search on Full Data (Option 2) ===", file=sys.stdout)
    pipeline = make_base_pipeline()
    param_grid = build_param_grid(random_state)
    full_cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state + 10)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=full_cv,
        scoring="f1_macro",
        n_jobs=-1,
        refit=True,
    )

    t0 = time.time()
    with tqdm(total=1, desc="Full data grid search", **tqdm_kwargs) as pbar:
        grid.fit(X, y)
        pbar.update(1)

    elapsed = time.time() - t0
    tqdm.write(
        f"Full-data search done in {elapsed:.1f}s | "
        f"Best model = {type(grid.best_estimator_.named_steps['model']).__name__}",
        file=sys.stdout,
    )
    tqdm.write(f"Best params: {grid.best_params_}", file=sys.stdout)
    tqdm.write(f"Cross-validated F1 on full data: {grid.best_score_:.3f}", file=sys.stdout)

    return grid


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def main(csv_path: str) -> None:
    X, y = load_dataset(csv_path)
    print(f"Loaded {len(X)} records with {X.shape[1]} features from {csv_path}")
    print("Class distribution:")
    for cls, count in y.value_counts().sort_index().items():
        print(f"   Class {cls}: {count}")

    with tqdm(total=3, desc="Workflow", **tqdm_kwargs) as pbar:
        # Step 1: Nested CV
        report = run_nested_cv(X, y)
        pbar.update(1)

        # Step 2: Full-data hyperparameter search
        final_grid = hyperparameter_search_full_data(X, y)
        pbar.update(1)

        # Step 3: Final model refit
        tqdm.write("\n=== Final Model Training ===", file=sys.stdout)
        final_model = final_grid.best_estimator_.fit(X, y)
        tqdm.write("Final pipeline ready for deployment.", file=sys.stdout)
        pbar.update(1)

    tqdm.write(
        f"\nReported generalisation (from nested CV): {report.outer_accuracy:.3f}\n"
        "Final model trained on all data using confirmed hyperparameters.\n",
        file=sys.stdout,
    )


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nested CV plus final model training.")
    parser.add_argument(
        "--csv",
        default="./Beerconsumption.csv",
        help="Path to the Beer consumption CSV file.",
    )
    args = parser.parse_args()
    main(args.csv)
