#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid search for MLP architectures in Experiment 2.
For each of the 6 architectures, tries different hyperparameter combinations
and reports the best config (val R², no energy measurement).

Usage:
  conda run -n python_cource python grid_search.py [--quick]
"""

import argparse
import itertools
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing import Preprocessor


# ─── 6 architectures ──────────────────────────────────────────
ARCHITECTURES = [
    (250, 30),
    (128, 64, 32),
    (32, 64, 128),
    (70, 70, 70),
    (90, 60, 50, 30),
    (40, 150, 40),
]


def load_data(data_path, test_size=0.2, random_state=42):
    df = pd.read_csv(data_path)
    target_col = 'median_house_value'
    y = df[target_col].values.reshape(-1, 1)
    X_df = df.drop(columns=[target_col])

    pre = Preprocessor()
    X_proc = pre.fit_transform(X_df)

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y).ravel()

    X_train, X_val, y_train, y_val = train_test_split(
        X_proc, y_scaled, test_size=test_size, random_state=random_state
    )
    return X_train, X_val, y_train, y_val


def run_trial(hidden, activation, lr, alpha, batch_size,
              max_epochs, patience, X_train, y_train, X_val, y_val, seed):
    """Train MLP and return best val R² and best epoch."""
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation=activation,
        solver='adam',
        learning_rate_init=lr,
        alpha=alpha,
        batch_size=batch_size,
        max_iter=1,
        warm_start=True,
        random_state=seed,
        verbose=False,
    )

    best_r2 = -np.inf
    no_improve = 0
    best_epoch = 0

    for epoch in range(1, max_epochs + 1):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)

        if r2 > best_r2 + 1e-6:
            best_r2 = r2
            no_improve = 0
            best_epoch = epoch
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best_r2, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/housing.csv")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer combos")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    proj_root = PROJECT_ROOT
    data_path = proj_root / args.data

    print("Loading data...")
    X_train, X_val, y_train, y_val = load_data(data_path, random_state=args.seed)
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Features: {X_train.shape[1]}")

    # ─── Hyperparameter grid ────────────────────────────────
    activations = ["tanh", "relu"]

    if args.quick:
        lrs = [0.001, 0.0005]
        alphas = [0.001, 0.0001]
        batch_sizes = [64, 128]
    else:
        lrs = [0.005, 0.001, 0.0005, 0.0001]
        alphas = [0.01, 0.001, 0.0001]
        batch_sizes = [32, 64, 128, 256]

    max_epochs = 500
    patience = 20

    # ─── Search ─────────────────────────────────────────────
    results = []

    for hidden in ARCHITECTURES:
        label = "x".join(str(h) for h in hidden)
        print(f"\n{'='*60}")
        print(f"  Architecture: ({label})  params={sum(hidden)}")
        print(f"{'='*60}")

        best_config = None
        best_r2 = -np.inf

        for activation, lr, alpha, batch_size in itertools.product(
            activations, lrs, alphas, batch_sizes
        ):
            t0 = time.time()
            val_r2, best_epoch = run_trial(
                hidden, activation, lr, alpha, batch_size,
                max_epochs, patience,
                X_train, y_train, X_val, y_val, args.seed
            )
            elapsed = time.time() - t0

            results.append({
                "architecture": label,
                "hidden": str(hidden),
                "activation": activation,
                "lr": lr,
                "alpha": alpha,
                "batch_size": batch_size,
                "val_r2": round(val_r2, 5),
                "best_epoch": best_epoch,
                "time_s": round(elapsed, 1),
            })

            marker = " <<<" if val_r2 > best_r2 else ""
            print(f"  {activation:5s} lr={lr:.4f} alpha={alpha:.6f} "
                  f"batch={batch_size:3d} -> R²={val_r2:.4f} "
                  f"epoch={best_epoch:3d}{marker}")

            if val_r2 > best_r2:
                best_r2 = val_r2
                best_config = (activation, lr, alpha, batch_size, best_epoch)

        print(f"  ── Best: {best_config[0]:5s} lr={best_config[1]:.4f} "
              f"alpha={best_config[2]:.6f} batch={best_config[3]} "
              f"R²={best_r2:.4f}")

    # ─── Output ─────────────────────────────────────────────
    df = pd.DataFrame(results)
    csv_path = Path.cwd() / "grid_search_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFull results saved to: {csv_path}")

    # Best per architecture
    best_idx = df.groupby("architecture")["val_r2"].idxmax()
    best_df = df.loc[best_idx, ["architecture", "activation", "lr",
                                 "alpha", "batch_size", "val_r2",
                                 "best_epoch"]].reset_index(drop=True)
    best_df.columns = ["Architecture", "Activation", "LR",
                       "Alpha", "Batch", "Best R²", "Best Epoch"]

    print("\n" + "="*70)
    print("  BEST CONFIG PER ARCHITECTURE")
    print("="*70)
    for _, row in best_df.iterrows():
        print(f"  ({row['Architecture']:>15s})  "
              f"{row['Activation']:5s}  "
              f"lr={row['LR']:.4f}  "
              f"alpha={row['Alpha']:.6f}  "
              f"batch={int(row['Batch']):3d}  "
              f"R²={row['Best R²']:.4f}  "
              f"epoch={int(row['Best Epoch'])}")

    summary_path = Path.cwd() / "grid_search_best.csv"
    best_df.to_csv(summary_path, index=False)
    print(f"\nBest configs saved to: {summary_path}")


if __name__ == "__main__":
    main()
