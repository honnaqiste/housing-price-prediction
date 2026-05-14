#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import EnergyTracker, setup_logging

setup_logging(log_to_file=True, log_level=logging.INFO)

def load_and_preprocess(data_path, test_size=0.2, random_state=42):
    """加载数据，删除派生特征，标准化特征和目标"""
    df = pd.read_csv(data_path)
    derived_cols = ['rooms_per_household', 'bedrooms_per_room', 'population_per_household']
    df = df.drop(columns=[c for c in derived_cols if c in df.columns])

    target = 'median_house_value' if 'median_house_value' in df.columns else df.columns[-1]
    y = df[target].values.reshape(-1, 1)
    X_df = df.drop(columns=[target])

    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X_df)

    y_scaler = StandardScaler()
    y_scaled = y_scaler.fit_transform(y).ravel()

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=random_state
    )
    # 原始尺度的验证集（用于计算最终指标）
    y_val_orig = y_scaler.inverse_transform(y_val.reshape(-1, 1)).ravel()
    return X_train, X_val, y_train, y_val, y_val_orig, X_scaler, y_scaler

def compute_information_gain(y_true, y_pred, baseline_pred=None):
    if baseline_pred is None:
        baseline_pred = np.full_like(y_true, np.mean(y_true))
    baseline_mse = mean_squared_error(y_true, baseline_pred)
    model_mse = mean_squared_error(y_true, y_pred)
    if model_mse <= 0 or baseline_mse <= 0:
        return 0.0
    return 0.5 * np.log(baseline_mse / model_mse)

def main():
    parser = argparse.ArgumentParser(description="Train MLP with energy tracking")
    parser.add_argument("--data", default="data/processed/housing_encoded.csv")
    parser.add_argument("--hidden-layers", type=str, default="100,50", help="Comma-separated hidden layer sizes")
    parser.add_argument("--max-iter", type=int, default=200, help="Maximum epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--idle-duration", type=float, default=10.0)
    args = parser.parse_args()

    proj_root = Path(__file__).parent.parent
    data_path = proj_root / args.data
    output_dir = proj_root / "models" / "mlp"

    X_train, X_val, y_train, y_val, y_val_orig, X_scaler, y_scaler = load_and_preprocess(data_path)

    # 原始尺度的基线
    baseline_pred_orig = np.full_like(y_val_orig, np.mean(y_val_orig))
    baseline_mse_orig = mean_squared_error(y_val_orig, baseline_pred_orig)
    logging.info(f"Baseline MSE (original): {baseline_mse_orig:.6f}")

    # 解析隐藏层
    hidden_layers = tuple(map(int, args.hidden_layers.split(',')))
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        learning_rate_init=args.learning_rate,
        max_iter=1,          # 每次调用只训练一个 epoch，以便记录能耗
        warm_start=True,     # 保留权重继续训练
        random_state=42,
        verbose=False
    )

    tracker = EnergyTracker(enable_background_removal=True, idle_duration=args.idle_duration, verbose=True)

    energy_records = []
    info_records = []

    with tracker:
        # 初始点（未训练）
        start_energy = tracker.get_current_energy()
        energy_records.append(start_energy)
        info_records.append(0.0)

        # 逐步训练每个 epoch
        for epoch in range(1, args.max_iter + 1):
            model.partial_fit(X_train, y_train)   # 只训练一个 epoch
            current_energy = tracker.get_current_energy()
            y_pred_scaled = model.predict(X_val)
            y_pred_orig = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            current_gain = compute_information_gain(y_val_orig, y_pred_orig, baseline_pred_orig)
            energy_records.append(current_energy)
            info_records.append(current_gain)

            if epoch % 20 == 0:
                logging.info(f"Epoch {epoch}: energy={current_energy:.3f} J, gain={current_gain:.4f} nats")

    net_energy = tracker.get_energy()
    logging.info(f"Training completed. Net energy: {net_energy:.3f} J")

    # 最终评估
    y_pred_scaled = model.predict(X_val)
    y_pred_orig = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    final_gain = compute_information_gain(y_val_orig, y_pred_orig, baseline_pred_orig)
    final_mse = mean_squared_error(y_val_orig, y_pred_orig)
    final_r2 = r2_score(y_val_orig, y_pred_orig)
    logging.info(f"Validation MSE: {final_mse:.2f}, R2: {final_r2:.4f}, Info gain: {final_gain:.4f} nats")

    # 保存模型和 scaler
    output_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, output_dir / "mlp_model.joblib")
    joblib.dump(X_scaler, output_dir / "mlp_scaler.joblib")
    joblib.dump(y_scaler, output_dir / "mlp_y_scaler.joblib")

    # 保存曲线数据
    curve_df = pd.DataFrame({"energy_joules": energy_records, "info_gain_nats": info_records})
    curve_df.to_csv(output_dir / "energy_curve.csv", index=False)

    # 绘图
    plt.figure(figsize=(8,6))
    plt.plot(energy_records, info_records, marker='o', markersize=3, linestyle='-')
    plt.xlabel("Net Energy (Joules)")
    plt.ylabel("Information Gain (nats)")
    plt.title(f"MLP ({hidden_layers}) - Energy vs Information Gain")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "energy_curve.png", dpi=150)
    logging.info(f"Curve saved to {output_dir}/energy_curve.png")

if __name__ == "__main__":
    main()