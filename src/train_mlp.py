#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import sys
import time
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


def load_and_preprocess(data_path, test_size=0.1, random_state=42):
    """
    加载数据，删除派生特征，划分训练/验证/测试集（6:2:2 或按传入比例）
    返回标准化后的数据及 scaler 对象。
    """
    df = pd.read_csv(data_path)
    derived_cols = ['rooms_per_household', 'bedrooms_per_room', 'population_per_household']
    df = df.drop(columns=[c for c in derived_cols if c in df.columns])

    target = 'median_house_value' if 'median_house_value' in df.columns else df.columns[-1]
    y = df[target].values.reshape(-1, 1)
    X = df.drop(columns=[target])

    # 先划分测试集（stratify 不适用于回归，直接用随机划分）
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # 再从剩余数据中划分训练集和验证集（验证集占剩余数据的 20% -> 总体 16% 左右）
    val_size = test_size / (1 - test_size)  # 使验证集占总体的比例约等于 test_size
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state
    )

    # 标准化特征
    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_val_scaled = X_scaler.transform(X_val)
    X_test_scaled = X_scaler.transform(X_test)

    # 标准化目标值
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train).ravel()
    y_val_scaled = y_scaler.transform(y_val).ravel()
    y_test_scaled = y_scaler.transform(y_test).ravel()

    # 保留原始尺度的验证集和测试集（用于计算最终指标）
    y_val_orig = y_val.ravel()   # 已经是原始值
    y_test_orig = y_test.ravel()

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train_scaled, y_val_scaled, y_test_scaled,
            y_val_orig, y_test_orig,
            X_scaler, y_scaler)


def compute_info_gain_from_r2(r2):
    """从 R² 计算信息增益（nats）"""
    if r2 >= 1.0:
        r2 = 0.999999
    return -0.5 * np.log(1 - r2)


def r2_to_info_gain_target(target_r2):
    """目标 R² 对应的信息增益阈值"""
    return compute_info_gain_from_r2(target_r2)


def main():
    parser = argparse.ArgumentParser(description="Train MLP with energy tracking and early stopping by R2 threshold")
    parser.add_argument("--data", default="data/processed/housing_encoded.csv",
                        help="Path to dataset (relative to project root)")
    parser.add_argument("--hidden-layers", type=str, default="100,50",
                        help="Comma-separated hidden layer sizes, e.g. '100,50'")
    parser.add_argument("--max-epochs", type=int, default=1000,
                        help="Maximum number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Initial learning rate for Adam")
    parser.add_argument("--target-r2", type=float, default=0.86,
                        help="Target validation R2 to stop training")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save results (CSV, PNG, metrics)")
    parser.add_argument("--test-size", type=float, default=0.1,
                        help="Proportion of data for test set (kept separate)")
    parser.add_argument("--idle-duration", type=float, default=10.0,
                        help="Seconds for background power measurement")
    parser.add_argument("--save-model", action="store_true",
                        help="Save the final model (not recommended for large experiments)")
    args = parser.parse_args()

    # 设置随机种子（影响数据划分、模型初始化、训练过程）
    np.random.seed(args.seed)

    # 构建路径
    proj_root = Path(__file__).parent.parent
    data_path = proj_root / args.data
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 配置日志（输出到控制台，文件由 setup_logging 处理）
    setup_logging(log_to_file=True, log_level=logging.INFO)

    logging.info(f"Experiment started. Output dir: {output_dir}")
    logging.info(f"Parameters: {vars(args)}")

    # 加载数据
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     y_val_orig, y_test_orig,
     X_scaler, y_scaler) = load_and_preprocess(
        data_path, test_size=args.test_size, random_state=args.seed
    )

    # 计算基线 MSE（基于验证集原始值）
    baseline_pred_orig = np.full_like(y_val_orig, np.mean(y_val_orig))
    baseline_mse_orig = mean_squared_error(y_val_orig, baseline_pred_orig)
    logging.info(f"Baseline MSE (validation, original scale): {baseline_mse_orig:.6f}")

    # 解析隐藏层结构
    hidden_layers = tuple(map(int, args.hidden_layers.split(',')))
    logging.info(f"MLP structure: {hidden_layers}")

    # 创建 MLP 回归器（warm_start=True 以便逐 epoch 训练）
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        learning_rate_init=args.learning_rate,
        max_iter=1,
        warm_start=True,
        random_state=args.seed,
        verbose=False
    )

    # 能量追踪器
    tracker = EnergyTracker(
        enable_background_removal=True,
        idle_duration=args.idle_duration,
        verbose=True,
        log_to_file=True  # 可以保持 True，但输出目录已有 metrics.json，日志文件可能重复
    )

    # 记录每个 epoch 的数据
    records = []  # 每个元素: (epoch, cumulative_energy, val_r2, info_gain)
    best_val_r2 = -np.inf
    best_epoch = 0
    stopped_early = False

    # 开始训练
    with tracker:
        start_energy = tracker.get_current_energy()
        records.append((0, start_energy, 0.0, 0.0))  # epoch 0: 初始状态

        for epoch in range(1, args.max_epochs + 1):
            # 训练一个 epoch
            model.partial_fit(X_train, y_train)

            # 获取当前累计净能耗
            cumulative_energy = tracker.get_current_energy()

            # 验证集预测（标准化尺度）
            y_val_pred_scaled = model.predict(X_val)
            # 逆变换到原始尺度
            y_val_pred_orig = y_scaler.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).ravel()

            # 计算验证集 R² 和信息增益
            val_r2 = r2_score(y_val_orig, y_val_pred_orig)
            # 信息增益: 0.5 * log(baseline_mse / model_mse)
            model_mse = mean_squared_error(y_val_orig, y_val_pred_orig)
            if model_mse > 0 and baseline_mse_orig > 0:
                info_gain = 0.5 * np.log(baseline_mse_orig / model_mse)
            else:
                info_gain = 0.0

            records.append((epoch, cumulative_energy, val_r2, info_gain))

            # 记录最佳 R²
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_epoch = epoch

            # 早停条件：验证 R² 达到目标阈值
            if val_r2 >= args.target_r2:
                logging.info(f"Target R2={args.target_r2} reached at epoch {epoch} (R2={val_r2:.4f})")
                stopped_early = True
                break

            # 定期输出日志
            if epoch % 20 == 0:
                logging.info(f"Epoch {epoch}: energy={cumulative_energy:.3f} J, R2={val_r2:.4f}, gain={info_gain:.4f} nats")

        if not stopped_early:
            logging.warning(f"Did NOT reach target R2={args.target_r2} after {args.max_epochs} epochs. "
                            f"Best R2={best_val_r2:.4f} at epoch {best_epoch}")

    # 训练结束，获取总净能耗（从进入 with 块到 exit）
    total_energy_net = tracker.get_energy()

    # 最终测试集评估（使用停止时或最后 epoch 的模型）
    y_test_pred_scaled = model.predict(X_test)
    y_test_pred_orig = y_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).ravel()
    test_r2 = r2_score(y_test_orig, y_test_pred_orig)
    test_mse = mean_squared_error(y_test_orig, y_test_pred_orig)

    # 最后验证集评估（与停止时一致）
    final_epoch = records[-1][0]
    final_val_r2 = records[-1][2]
    final_info_gain = records[-1][3]

    logging.info(f"Training finished. Total net energy: {total_energy_net:.3f} J")
    logging.info(f"Stopped at epoch {final_epoch} (early stop: {stopped_early})")
    logging.info(f"Validation R2: {final_val_r2:.4f}, Info gain: {final_info_gain:.4f} nats")
    logging.info(f"Test R2: {test_r2:.4f}, Test MSE: {test_mse:.2f}")

    # 保存能耗曲线 CSV
    df_curve = pd.DataFrame(records, columns=["epoch", "cumulative_energy_j", "val_r2", "info_gain_nats"])
    curve_csv = output_dir / "energy_curve.csv"
    df_curve.to_csv(curve_csv, index=False)
    logging.info(f"Energy curve saved to {curve_csv}")

    # 保存指标 JSON（包含背景功率）
    metrics = {
        "structure": args.hidden_layers,
        "seed": args.seed,
        "target_r2": args.target_r2,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "test_size": args.test_size,
        "stopped_early": stopped_early,
        "final_epoch": final_epoch,
        "total_energy_j": total_energy_net,
        "best_val_r2": best_val_r2,
        "best_epoch": best_epoch,
        "final_val_r2": final_val_r2,
        "final_info_gain": final_info_gain,
        "test_r2": test_r2,
        "test_mse": test_mse,
        "baseline_mse": baseline_mse_orig,
        # 新增背景功耗信息
        "background_power_watts": tracker.background_power_watts if hasattr(tracker, 'background_power_watts') else None,
        "idle_duration": args.idle_duration,
    }
    metrics_json = output_dir / "metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Metrics saved to {metrics_json}")

    # 可选：绘制能耗 vs 信息增益曲线
    plt.figure(figsize=(8, 6))
    plt.plot(df_curve["cumulative_energy_j"], df_curve["info_gain_nats"], marker='o', markersize=3, linestyle='-')
    plt.xlabel("Net Energy (Joules)")
    plt.ylabel("Information Gain (nats)")
    plt.title(f"MLP {hidden_layers} (seed={args.seed})")
    plt.grid(True, alpha=0.3)
    # 在图上标记停止点
    stop_energy = df_curve[df_curve["epoch"] == final_epoch]["cumulative_energy_j"].values[0]
    stop_gain = df_curve[df_curve["epoch"] == final_epoch]["info_gain_nats"].values[0]
    plt.scatter(stop_energy, stop_gain, color='red', s=80, label=f'Stop (R2={final_val_r2:.3f})')
    plt.legend()
    curve_png = output_dir / "energy_curve.png"
    plt.savefig(curve_png, dpi=150)
    logging.info(f"Plot saved to {curve_png}")

    if args.save_model:
        import joblib
        model_path = output_dir / "model.joblib"
        joblib.dump(model, model_path)
        logging.info(f"Model saved to {model_path} (not recommended for large experiments)")

    logging.info("Experiment finished.")


if __name__ == "__main__":
    main()