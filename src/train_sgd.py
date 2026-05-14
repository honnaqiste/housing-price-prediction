#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/housing_encoded.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--idle-duration", type=float, default=10.0)
    args = parser.parse_args()

    proj_root = Path(__file__).parent.parent
    data_path = proj_root / args.data
    output_dir = proj_root / "models" / "sgd"

    X_train, X_val, y_train, y_val, y_val_orig, X_scaler, y_scaler = load_and_preprocess(data_path)

    # 原始尺度的基线
    baseline_pred_orig = np.full_like(y_val_orig, np.mean(y_val_orig))
    baseline_mse_orig = mean_squared_error(y_val_orig, baseline_pred_orig)
    logging.info(f"Baseline MSE (orig): {baseline_mse_orig:.6f}")

    tracker = EnergyTracker(enable_background_removal=True, idle_duration=args.idle_duration, verbose=True)

    model = SGDRegressor(loss='squared_error', max_iter=1, tol=None,
                         eta0=args.lr, learning_rate='invscaling', random_state=42)

    energy_records = []
    info_records = []

    with tracker:
        # 初始点
        start_energy = tracker.get_current_energy()
        energy_records.append(start_energy)
        info_records.append(0.0)

        for epoch in range(1, args.epochs + 1):
            model.partial_fit(X_train, y_train)
            # 记录当前累计能量
            current_energy = tracker.get_current_energy()
            # 预测（标准化尺度）并反标准化到原始尺度
            y_pred_scaled = model.predict(X_val)
            y_pred_orig = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            current_gain = compute_information_gain(y_val_orig, y_pred_orig, baseline_pred_orig)
            energy_records.append(current_energy)
            info_records.append(current_gain)

            if epoch % 10 == 0:
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

    # 保存模型、scaler、曲线
    output_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, output_dir / "sgd_model.joblib")
    joblib.dump(X_scaler, output_dir / "sgd_scaler.joblib")
    joblib.dump(y_scaler, output_dir / "sgd_y_scaler.joblib")

    curve_df = pd.DataFrame({"energy_joules": energy_records, "info_gain_nats": info_records})
    curve_df.to_csv(output_dir / "energy_curve.csv", index=False)

    # 绘图
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(energy_records, info_records, marker='o')
    plt.xlabel("Net Energy (J)")
    plt.ylabel("Information Gain (nats)")
    plt.title("SGD Training: Energy vs Information Gain")
    plt.grid(True)
    plt.savefig(output_dir / "energy_curve.png", dpi=150)
    logging.info(f"Curve saved to {output_dir}/energy_curve.png")

if __name__ == "__main__":
    main()