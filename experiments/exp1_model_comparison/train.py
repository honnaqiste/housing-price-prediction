#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 1: Cross-model energy efficiency comparison.
Unified training script with user-specified hyperparameters.

Usage:
  # With energy (needs root + RAPL):
  taskset -c 0 python train.py --model linear
  taskset -c 0 python train.py --model random_forest
  # No energy (any user, for hyperparameter search):
  python train.py --model mlp --mlp-hidden "128,64,32" --no-energy
  python train.py --model histgb --no-energy
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils import EnergyTracker, setup_logging
from src.preprocessing import Preprocessor

setup_logging(log_to_file=True, log_level=logging.INFO)


def compute_information_gain(y_true, y_pred, baseline_pred=None):
    """0.5 * log(baseline_MSE / model_MSE)  in nats."""
    if baseline_pred is None:
        baseline_pred = np.full_like(y_true, np.mean(y_true))
    baseline_mse = mean_squared_error(y_true, baseline_pred)
    model_mse = mean_squared_error(y_true, y_pred)
    if model_mse <= 0 or baseline_mse <= 0:
        return 0.0
    return 0.5 * np.log(baseline_mse / model_mse)


def info_gain_from_r2(r2):
    """Convert R² to information gain: -0.5 * log(1 - R²)."""
    r2 = np.clip(r2, None, 0.9999)
    return -0.5 * np.log(1 - r2)


class NoopTracker:
    """Drop-in for EnergyTracker when --no-energy is used."""
    background_power_watts = 0.0

    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def get_current_energy(self):
        return 0.0
    def get_energy(self):
        return 0.0


def load_data(data_path, test_size=0.2, random_state=42):
    df = pd.read_csv(data_path)
    target_col = 'median_house_value'
    y = df[target_col].values.reshape(-1, 1)
    X_df = df.drop(columns=[target_col])

    pre = Preprocessor()
    X_proc = pre.fit_transform(X_df)
    logging.info(f"Feature matrix: {X_proc.shape} columns")

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y).ravel()

    X_train, X_val, y_train, y_val = train_test_split(
        X_proc, y_scaled, test_size=test_size, random_state=random_state
    )
    logging.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
    return X_train, X_val, y_train, y_val, pre, y_scaler


# ─── Model trainers ───────────────────────────────────────────


def train_linear(X_train, y_train, X_val, y_val, tracker, random_state):
    model = LinearRegression()
    energy_curve = []

    with tracker:
        e0 = tracker.get_current_energy()
        energy_curve.append({"step": 0, "cumulative_energy_j": e0,
                             "val_r2": 0.0, "info_gain_nats": 0.0})
        model.fit(X_train, y_train)
        e1 = tracker.get_current_energy()
        y_pred = model.predict(X_val)
        r2_val = r2_score(y_val, y_pred)
        gain = compute_information_gain(y_val, y_pred)
        energy_curve.append({"step": 1, "cumulative_energy_j": e1,
                             "val_r2": r2_val, "info_gain_nats": gain})

    logging.info(f"  LR: energy={e1:.3f}J, R2={r2_val:.4f}, gain={gain:.4f}")
    return model, energy_curve, r2_val, gain


def train_random_forest(X_train, y_train, X_val, y_val, tracker,
                        n_estimators=200, step_size=10, max_depth=None,
                        min_samples_split=2, min_samples_leaf=2,
                        max_features=None, random_state=42):
    model = RandomForestRegressor(
        warm_start=True, n_estimators=step_size,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=1, random_state=random_state
    )
    energy_curve = []

    with tracker:
        e0 = tracker.get_current_energy()
        energy_curve.append({"step": 0, "cumulative_energy_j": e0,
                             "val_r2": 0.0, "info_gain_nats": 0.0})
        for n in range(step_size, n_estimators + 1, step_size):
            model.set_params(n_estimators=n)
            model.fit(X_train, y_train)
            cum_e = tracker.get_current_energy()
            y_pred = model.predict(X_val)
            r2_val = r2_score(y_val, y_pred)
            gain = compute_information_gain(y_val, y_pred)
            energy_curve.append({"step": n, "cumulative_energy_j": cum_e,
                                 "val_r2": r2_val, "info_gain_nats": gain})
            logging.info(f"  RF trees={n}: energy={cum_e:.3f}J, R2={r2_val:.4f}")

    f = energy_curve[-1]
    return model, energy_curve, f["val_r2"], f["info_gain_nats"]


def train_histgb(X_train, y_train, X_val, y_val, tracker,
                 max_iter=1000, step_size=10, max_depth=10,
                 max_bins=255, learning_rate=0.05,
                 l2_regularization=1.0, random_state=42):
    model = HistGradientBoostingRegressor(
        warm_start=True, max_iter=step_size,
        max_depth=max_depth, max_bins=max_bins,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        random_state=random_state
    )
    energy_curve = []

    with tracker:
        e0 = tracker.get_current_energy()
        energy_curve.append({"step": 0, "cumulative_energy_j": e0,
                             "val_r2": 0.0, "info_gain_nats": 0.0})
        for cur in range(step_size, max_iter + 1, step_size):
            model.set_params(max_iter=cur)
            model.fit(X_train, y_train)
            cum_e = tracker.get_current_energy()
            y_pred = model.predict(X_val)
            r2_val = r2_score(y_val, y_pred)
            gain = compute_information_gain(y_val, y_pred)
            energy_curve.append({"step": cur, "cumulative_energy_j": cum_e,
                                 "val_r2": r2_val, "info_gain_nats": gain})
            logging.info(f"  HistGB iter={cur}: energy={cum_e:.3f}J, R2={r2_val:.4f}")

    f = energy_curve[-1]
    return model, energy_curve, f["val_r2"], f["info_gain_nats"]


def train_mlp(X_train, y_train, X_val, y_val, tracker,
              hidden_layer_sizes=(256, 128, 64), max_epochs=500, patience=20,
              learning_rate_init=0.0005, activation='tanh',
              alpha=0.001, batch_size=128, random_state=42):
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver='adam',
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        batch_size=batch_size,
        max_iter=1, warm_start=True,
        random_state=random_state, verbose=False
    )
    energy_curve = []
    best_val_r2 = -np.inf
    no_improve = 0
    best_epoch = 0
    best_weights = None

    with tracker:
        e0 = tracker.get_current_energy()
        energy_curve.append({"step": 0, "cumulative_energy_j": e0,
                             "val_r2": 0.0, "info_gain_nats": 0.0})
        for epoch in range(1, max_epochs + 1):
            model.fit(X_train, y_train)
            cum_e = tracker.get_current_energy()
            y_pred = model.predict(X_val)
            r2_val = r2_score(y_val, y_pred)
            gain = compute_information_gain(y_val, y_pred)
            energy_curve.append({"step": epoch, "cumulative_energy_j": cum_e,
                                 "val_r2": r2_val, "info_gain_nats": gain})

            if r2_val > best_val_r2 + 1e-6:
                best_val_r2 = r2_val
                no_improve = 0
                best_epoch = epoch
                best_weights = {
                    "coefs": [c.copy() for c in model.coefs_],
                    "intercepts": [i.copy() for i in model.intercepts_]
                }
            else:
                no_improve += 1

            if epoch % 50 == 0:
                logging.info(f"  MLP epoch={epoch}: energy={cum_e:.3f}J, "
                             f"R2={r2_val:.4f} (best={best_val_r2:.4f})")
            if no_improve >= patience:
                logging.info(f"  Early stop epoch={epoch}, best={best_epoch} "
                             f"(R2={best_val_r2:.4f})")
                break

    if best_weights:
        model.coefs_ = best_weights["coefs"]
        model.intercepts_ = best_weights["intercepts"]
        model.n_iter_ = best_epoch

    y_pred = model.predict(X_val)
    final_r2 = r2_score(y_val, y_pred)
    final_gain = compute_information_gain(y_val, y_pred)

    logging.info(f"  MLP final: energy={tracker.get_energy():.3f}J, "
                 f"R2={final_r2:.4f}, gain={final_gain:.4f}, best_epoch={best_epoch}")
    return model, energy_curve, final_r2, final_gain, best_epoch


# ─── main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 1: Model energy efficiency")
    parser.add_argument("--data", type=str, default="data/raw/housing.csv")
    parser.add_argument("--model", type=str, required=True,
                        choices=["linear", "random_forest", "histgb", "mlp"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--idle-duration", type=float, default=10.0)
    parser.add_argument("--no-energy", action="store_true",
                        help="Skip RAPL energy measurement (no root needed)")

    # Shared
    parser.add_argument("--step-size", type=int, default=10)

    # RF
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--rf-step-size", type=int, default=5)
    parser.add_argument("--rf-depth", type=int, default=None)
    parser.add_argument("--rf-min-samples-split", type=int, default=2)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--rf-max-features", type=str, default=None)

    # HistGB
    parser.add_argument("--histgb-iter", type=int, default=1000)
    parser.add_argument("--histgb-depth", type=int, default=10)
    parser.add_argument("--histgb-bins", type=int, default=255)
    parser.add_argument("--histgb-lr", type=float, default=0.05)
    parser.add_argument("--histgb-l2", type=float, default=1.0)

    # MLP
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--mlp-hidden", type=str, default="256,128,64")
    parser.add_argument("--mlp-lr", type=float, default=0.0005)
    parser.add_argument("--mlp-activation", type=str, default="tanh",
                        choices=["relu", "tanh", "logistic"])
    parser.add_argument("--mlp-alpha", type=float, default=0.001)
    parser.add_argument("--mlp-batch-size", type=int, default=128)

    args = parser.parse_args()

    proj_root = PROJECT_ROOT
    data_path = proj_root / args.data
    model_tag = args.model
    out_dir = Path(args.output_dir) if args.output_dir else \
        proj_root / "experiments" / "exp1_model_comparison" / "results" / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"=== Experiment 1: {model_tag} ===")
    logging.info(f"Data: {data_path}")
    logging.info(f"Output: {out_dir}")

    X_train, X_val, y_train, y_val, pre, y_scaler = load_data(
        data_path, test_size=args.test_size, random_state=args.seed
    )

    baseline_pred = np.full_like(y_val, np.mean(y_val))
    baseline_mse = mean_squared_error(y_val, baseline_pred)
    logging.info(f"Baseline MSE (scaled): {baseline_mse:.4f}")

    if args.no_energy:
        logging.info("No-energy mode (using NoopTracker)")
        tracker = NoopTracker()
    else:
        tracker = EnergyTracker(enable_background_removal=True,
                                idle_duration=args.idle_duration,
                                verbose=True)

    start_wall = time.time()

    if args.model == "linear":
        model, ec, r2, gain = train_linear(
            X_train, y_train, X_val, y_val, tracker, args.seed)
    elif args.model == "random_forest":
        mf = args.rf_max_features
        if mf is not None and mf.lower() in ("none", "null"):
            mf = None
        model, ec, r2, gain = train_random_forest(
            X_train, y_train, X_val, y_val, tracker,
            n_estimators=args.rf_trees, step_size=args.rf_step_size,
            max_depth=args.rf_depth,
            min_samples_split=args.rf_min_samples_split,
            min_samples_leaf=args.rf_min_samples_leaf,
            max_features=mf, random_state=args.seed)
    elif args.model == "histgb":
        model, ec, r2, gain = train_histgb(
            X_train, y_train, X_val, y_val, tracker,
            max_iter=args.histgb_iter, step_size=args.step_size,
            max_depth=args.histgb_depth, max_bins=args.histgb_bins,
            learning_rate=args.histgb_lr,
            l2_regularization=args.histgb_l2, random_state=args.seed)
    elif args.model == "mlp":
        hidden = tuple(map(int, args.mlp_hidden.split(',')))
        model, ec, r2, gain, best_epoch = train_mlp(
            X_train, y_train, X_val, y_val, tracker,
            hidden_layer_sizes=hidden,
            max_epochs=args.max_epochs, patience=args.patience,
            learning_rate_init=args.mlp_lr,
            activation=args.mlp_activation,
            alpha=args.mlp_alpha,
            batch_size=args.mlp_batch_size,
            random_state=args.seed)

    wall_time = time.time() - start_wall
    net_energy = tracker.get_energy()
    eff = gain / max(net_energy, 0.001)

    logging.info(f"=== {model_tag} done ===")
    logging.info(f"Wall time: {wall_time:.1f}s")
    logging.info(f"Net energy: {net_energy:.3f} J")
    logging.info(f"Final val R2: {r2:.4f}")
    logging.info(f"Final info gain: {gain:.4f} nats")
    logging.info(f"Efficiency: {eff:.4f} nats/J")

    metrics = {
        "model": model_tag,
        "mlp_hidden": args.mlp_hidden if args.model == "mlp" else None,
        "seed": args.seed,
        "test_size": args.test_size,
        "total_energy_j": net_energy,
        "wall_time_s": round(wall_time, 2),
        "final_val_r2": r2,
        "final_info_gain": gain,
        "baseline_mse": baseline_mse,
        "background_power_watts": tracker.background_power_watts,
        "energy_efficiency_nats_per_j": round(eff, 6),
        "model_params": {k: v for k, v in vars(args).items()
                         if k not in ("data", "model", "output_dir")},
    }
    if args.model == "mlp":
        metrics["best_epoch"] = best_epoch

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Saved: {out_dir / 'metrics.json'}")

    pd.DataFrame(ec).to_csv(out_dir / "energy_curve.csv", index=False)
    logging.info(f"Saved: {out_dir / 'energy_curve.csv'}")

    import joblib
    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(pre, out_dir / "preprocessor.joblib")
    joblib.dump(y_scaler, out_dir / "y_scaler.joblib")

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        df_c = pd.DataFrame(ec)
        plt.figure(figsize=(8, 5))
        plt.plot(df_c["cumulative_energy_j"], df_c["info_gain_nats"],
                 marker='.', linestyle='-', linewidth=1.5)
        plt.xlabel("Cumulative Energy (J)")
        plt.ylabel("Information Gain (nats)")
        plt.title(f"{model_tag} -- Energy-Information Curve")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "energy_curve.png", dpi=150)
        plt.close()
        logging.info(f"Saved: {out_dir / 'energy_curve.png'}")
    except Exception as e:
        logging.warning(f"Plot skipped: {e}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
