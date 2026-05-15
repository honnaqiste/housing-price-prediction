#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析不同 MLP 结构首次达到指定 R² 阈值（0.5, 0.75）时的能量消耗。
遍历实验目录下所有 exp_*/run*/energy_curve.csv，结合 metrics.json，
生成汇总表并绘制对比图表。
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 忽略 pandas 未来警告
warnings.filterwarnings("ignore", category=FutureWarning)

# ================== 配置 ==================
THRESHOLDS = [0.5, 0.75]          # 要分析的 R² 阈值
EXPERIMENT_ROOT = Path(".")       # 当前目录应为实验根目录 (20260515)
OUTPUT_CSV = "threshold_energy_summary.csv"
OUTPUT_PLOT = "threshold_energy_boxplot.png"

# ================== 辅助函数 ==================
def find_first_threshold_energy(csv_path, threshold, energy_col="cumulative_energy_j", r2_col="val_r2"):
    """
    从 energy_curve.csv 中查找首次达到指定 R² 阈值时的累积能量。
    如果从未达到，返回 NaN。
    """
    if not csv_path.exists():
        return np.nan
    try:
        df = pd.read_csv(csv_path)
        # 确保列存在
        if energy_col not in df.columns or r2_col not in df.columns:
            return np.nan
        # 找到第一个满足 R² >= threshold 的行
        mask = df[r2_col] >= threshold
        if mask.any():
            first_idx = mask.idxmax()  # 第一个 True 的索引
            return df.loc[first_idx, energy_col]
        else:
            return np.nan
    except Exception as e:
        print(f"警告: 读取 {csv_path} 出错: {e}")
        return np.nan

# ================== 主程序 ==================
def main():
    root = Path(EXPERIMENT_ROOT)
    if not root.exists():
        print(f"错误: 目录 {root.absolute()} 不存在")
        return

    # 收集所有 metrics.json 文件（代表一次成功训练）
    metrics_files = list(root.glob("exp_*/run*/metrics.json"))
    if not metrics_files:
        print("未找到任何 metrics.json 文件，请确认实验目录结构正确。")
        return

    records = []
    missing_curves = 0

    for mf in metrics_files:
        run_dir = mf.parent
        exp_dir = run_dir.parent
        # 解析结构名称: exp_01_50 -> 50 ; exp_04_50_50 -> 50,50
        parts = exp_dir.name.split('_', 2)
        if len(parts) >= 3:
            structure = parts[2].replace('_', ',')
        else:
            structure = exp_dir.name
        run_id = int(run_dir.name[3:])

        # 读取 metrics.json
        with open(mf, 'r') as f:
            meta = json.load(f)

        # 能量曲线文件路径
        curve_path = run_dir / "energy_curve.csv"
        if not curve_path.exists():
            print(f"警告: 缺少曲线文件 {curve_path}")
            missing_curves += 1
            # 仍然记录，但阈值为 NaN
            energy_05 = energy_075 = np.nan
        else:
            energy_05 = find_first_threshold_energy(curve_path, 0.5)
            energy_075 = find_first_threshold_energy(curve_path, 0.75)

        record = {
            "structure": structure,
            "run_id": run_id,
            "seed": meta.get("seed"),
            "stopped_early": meta.get("stopped_early", False),
            "total_energy_j": meta.get("total_energy_j", np.nan),
            "best_val_r2": meta.get("best_val_r2", np.nan),
            "test_r2": meta.get("test_r2", np.nan),
            "background_power_watts": meta.get("background_power_watts", np.nan),
            "energy_to_0.5": energy_05,
            "energy_to_0.75": energy_075,
        }
        records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        print("没有有效记录，退出。")
        return

    # 排序
    df = df.sort_values(["structure", "run_id"])

    # 统计缺失情况
    missing_05 = df["energy_to_0.5"].isna().sum()
    missing_075 = df["energy_to_0.75"].isna().sum()
    print(f"总运行次数: {len(df)}")
    print(f"缺少曲线文件: {missing_curves}")
    print(f"从未达到 R²=0.5 的运行数: {missing_05}")
    print(f"从未达到 R²=0.75 的运行数: {missing_075}")

    # 保存完整汇总
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"详细汇总已保存至: {OUTPUT_CSV}")

    # ================== 可视化 ==================
    # 只绘制达到阈值的 run（排除 NaN）
    plot_data_05 = df[df["energy_to_0.5"].notna()].copy()
    plot_data_075 = df[df["energy_to_0.75"].notna()].copy()

    if plot_data_05.empty and plot_data_075.empty:
        print("没有足够数据绘制箱线图。")
        return

    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # 调整布局
    if not plot_data_05.empty:
        sns.boxplot(data=plot_data_05, x="structure", y="energy_to_0.5", ax=axes[0])
        axes[0].set_title("Energy to reach R² = 0.5")
        axes[0].set_xlabel("Structure")
        axes[0].set_ylabel("Cumulative Energy (Joules)")
        axes[0].tick_params(axis='x', rotation=45)
    else:
        axes[0].text(0.5, 0.5, "No data for R²=0.5", ha='center', va='center')
        axes[0].set_title("R²=0.5")
    if not plot_data_075.empty:
        sns.boxplot(data=plot_data_075, x="structure", y="energy_to_0.75", ax=axes[1])
        axes[1].set_title("Energy to reach R² = 0.75")
        axes[1].set_xlabel("Structure")
        axes[1].set_ylabel("Cumulative Energy (Joules)")
        axes[1].tick_params(axis='x', rotation=45)
    else:
        axes[1].text(0.5, 0.5, "No data for R²=0.75", ha='center', va='center')
        axes[1].set_title("R²=0.75")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150)
    print(f"箱线图已保存至: {OUTPUT_PLOT}")

    # 额外统计：最佳 R² 与达到 0.75 能量的散点图
    if not plot_data_075.empty:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=plot_data_075, x="energy_to_0.75", y="best_val_r2", hue="structure", style="structure")
        plt.xlabel("Energy to reach R²=0.75 (Joules)")
        plt.ylabel("Best validation R²")
        plt.title("Best R² vs Energy to reach 0.75")
        plt.grid(True, alpha=0.3)
        scatter_plot = "best_r2_vs_energy_0.75.png"
        plt.savefig(scatter_plot, dpi=150)
        print(f"散点图已保存至: {scatter_plot}")

    print("分析完成。")

if __name__ == "__main__":
    main()