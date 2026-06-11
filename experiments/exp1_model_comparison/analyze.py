#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 1: Cross-model energy efficiency comparison — Analysis.
Scans results/{model}/run_{N}/metrics.json + energy_curve.csv.
Generates:
  - Summary table (per run and aggregated)
  - Energy threshold analysis: energy to reach R2=0.5, 0.75, 0.785
  - Bar chart of threshold energy by model
  - Scatter plot: total energy vs info gain (Pareto frontier)
  - Energy curves overlay with threshold markers

Usage:
  conda run -n python_cource python analyze.py
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Model metadata for display ───────────────────────────────
MODEL_META = {
    "linear": {
        "label": "LinearRegression",
        "marker": "s", "color": "#2196F3", "linestyle": "-"
    },
    "random_forest": {
        "label": "RandomForest (200)",
        "marker": "o", "color": "#4CAF50", "linestyle": "-"
    },
    "histgb": {
        "label": "HistGB (1000 iter)",
        "marker": "^", "color": "#FF9800", "linestyle": "-"
    },
    "mlp_256_128_64": {
        "label": "MLP (256,128,64)",
        "marker": "D", "color": "#E91E63", "linestyle": "-"
    },
    "mlp_128_64_32": {
        "label": "MLP (128,64,32)",
        "marker": "v", "color": "#9C27B0", "linestyle": "--"
    },
    "mlp_128_64": {
        "label": "MLP (128,64)",
        "marker": "p", "color": "#00BCD4", "linestyle": ":"
    },
}

# R2 thresholds to analyse
R2_THRESHOLDS = [0.5, 0.75, 0.79]
# Corresponding info gain: IG = -0.5 * log(1 - R2)
IG_THRESHOLDS = [-0.5 * np.log(max(1 - r, 1e-10)) for r in R2_THRESHOLDS]


def find_first_above(curve_df, col, threshold):
    """Return the first cumulative_energy_j where col >= threshold."""
    mask = curve_df[col] >= threshold
    if mask.any():
        idx = mask.idxmax()
        return curve_df.loc[idx, "cumulative_energy_j"]
    return np.nan


def aggregate_runs(results_root):
    """Return dict: model_name -> list of run metrics dicts."""
    all_data = {}
    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        runs = []
        for run_dir in sorted(model_dir.glob("run_*/")):
            mf = run_dir / "metrics.json"
            cf = run_dir / "energy_curve.csv"
            if not mf.exists():
                continue
            with open(mf) as f:
                m = json.load(f)

            m["_run_id"] = run_dir.name
            m["_model_dir"] = model_name

            # Threshold analysis from energy curve
            if cf.exists():
                curve = pd.read_csv(cf)
                curve["cumulative_energy_j"] = curve["cumulative_energy_j"].clip(lower=0)
                for th in R2_THRESHOLDS:
                    key = f"energy_to_r2_{th}"
                    m[key] = find_first_above(curve, "val_r2", th)
                for ig in IG_THRESHOLDS:
                    key = f"energy_to_ig_{ig:.3f}"
                    m[key] = find_first_above(curve, "info_gain_nats", ig)

            runs.append(m)
        if runs:
            all_data[model_name] = runs
    return all_data


def pareto_frontier(points):
    """Return bool mask of Pareto-optimal points (min energy, max gain)."""
    pts = np.array(points)
    n = len(pts)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (pts[j, 0] <= pts[i, 0] and pts[j, 1] >= pts[i, 1] and
                    (pts[j, 0] < pts[i, 0] or pts[j, 1] > pts[i, 1])):
                is_pareto[i] = False
                break
    return is_pareto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    results_root = Path(args.results_dir).resolve() if args.results_dir else \
        script_dir / "results"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else script_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results: {results_root}")
    all_data = aggregate_runs(results_root)

    if not all_data:
        print("No results found.")
        return

    print(f"Found models: {list(all_data.keys())}")

    # ── Build aggregated table ───────────────────────────────
    rows = []
    for model_name, runs in all_data.items():
        energies = [r["total_energy_j"] for r in runs]
        gains = [r["final_info_gain"] for r in runs]
        r2s = [r["final_val_r2"] for r in runs]
        effs = [r.get("energy_efficiency_nats_per_j", 0) for r in runs]

        row = {
            "model": model_name,
            "n_runs": len(runs),
            "energy_mean": np.mean(energies),
            "energy_std": np.std(energies, ddof=1) if len(energies) > 1 else 0,
            "gain_mean": np.mean(gains),
            "r2_mean": np.mean(r2s),
            "eff_mean": np.mean(effs),
        }
        # Threshold means
        for th in R2_THRESHOLDS:
            key = f"energy_to_r2_{th}"
            vals = [r.get(key, np.nan) for r in runs]
            vals = [v for v in vals if not np.isnan(v)]
            row[f"e_r2_{th}_mean"] = np.mean(vals) if vals else np.nan

        rows.append(row)

    df_summary = pd.DataFrame(rows)
    print("\n=== Aggregated Results ===")
    print(df_summary.to_string(index=False))

    # Save
    df_summary.to_csv(output_dir / "exp1_summary.csv", index=False)
    print(f"\nSaved: {output_dir / 'exp1_summary.csv'}")

    # ── Per-run detailed CSV ─────────────────────────────────
    detail_rows = []
    for model_name, runs in all_data.items():
        for r in runs:
            dr = {
                "model": model_name,
                "run": r["_run_id"],
                "seed": r["seed"],
                "energy_j": r["total_energy_j"],
                "val_r2": r["final_val_r2"],
                "info_gain": r["final_info_gain"],
                "best_epoch": r.get("best_epoch", ""),
            }
            for th in R2_THRESHOLDS:
                dr[f"e_to_r2_{th}"] = r.get(f"energy_to_r2_{th}", np.nan)
            detail_rows.append(dr)
    pd.DataFrame(detail_rows).to_csv(output_dir / "exp1_detail.csv", index=False)
    print(f"Saved: {output_dir / 'exp1_detail.csv'}")

    # ── Figure 1: Threshold Energy Bar Chart ─────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=False)

    for idx, (th, ig) in enumerate(zip(R2_THRESHOLDS, IG_THRESHOLDS)):
        ax = axes[idx]
        col = f"e_r2_{th}_mean"
        models = []
        means = []
        colors = []
        for _, row in df_summary.sort_values("model").iterrows():
            meta = MODEL_META.get(row["model"], {})
            v = row.get(col, np.nan)
            if not np.isnan(v) and v > 0:
                models.append(meta.get("label", row["model"]))
                means.append(v)
                colors.append(meta.get("color", "gray"))

        bars = ax.barh(range(len(models)), means, color=colors, height=0.6)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models, fontsize=10)
        ax.set_xlabel("Energy (J)")
        ax.set_title(f"R² ≥ {th}  (IG ≥ {ig:.3f} nats)", fontsize=12)
        ax.grid(True, axis='x', alpha=0.3)

        # Value labels
        for i, (bar, v) in enumerate(zip(bars, means)):
            ax.text(v + max(means)*0.01, i, f"{v:.1f}J",
                    va='center', fontsize=9)

    fig.suptitle("Energy Required to Reach R² Thresholds", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "exp1_threshold_energy.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {output_dir / 'exp1_threshold_energy.png'}")
    plt.close(fig)

    # ── Figure 2: Energy vs Info Gain (Pareto scatter) ──────
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    all_points = []

    for model_name in sorted(all_data.keys()):
        runs = all_data[model_name]
        meta = MODEL_META.get(model_name, {"label": model_name, "marker": "o",
                                            "color": "gray", "linestyle": "-"})
        energies = [r["total_energy_j"] for r in runs]
        gains = [r["final_info_gain"] for r in runs]
        e_mean, e_std = np.mean(energies), np.std(energies, ddof=1)
        g_mean, g_std = np.mean(gains), np.std(gains, ddof=1)

        # Individual points
        ax2.scatter(energies, gains, marker=meta["marker"], color=meta["color"],
                    s=30, alpha=0.25, edgecolors='none', zorder=2)

        # Mean
        ax2.scatter(e_mean, g_mean, marker=meta["marker"], color=meta["color"],
                    s=180, edgecolors='white', linewidth=1.5, zorder=6,
                    label=f'{meta["label"]} ({e_mean:.0f}J, {g_mean:.2f}nats)')
        ax2.errorbar(e_mean, g_mean, xerr=e_std, yerr=g_std,
                     fmt='none', ecolor=meta["color"], alpha=0.4, capsize=4)

        for e, g in zip(energies, gains):
            all_points.append((e, g))

    # Pareto
    is_pareto = pareto_frontier(all_points)
    pareto_pts = np.array([p for i, p in enumerate(all_points) if is_pareto[i]])
    if len(pareto_pts) > 1:
        pareto_pts = pareto_pts[np.argsort(pareto_pts[:, 0])]
        ax2.plot(pareto_pts[:, 0], pareto_pts[:, 1],
                 '--', color='gray', alpha=0.6, linewidth=1.5,
                 label="Pareto frontier")

    # Threshold lines
    for th, ig in zip(R2_THRESHOLDS, IG_THRESHOLDS):
        ax2.axhline(y=ig, color='red', alpha=0.15, linewidth=0.8)
        ax2.text(ax2.get_xlim()[1]*1.01, ig, f"R²={th}",
                 color='red', alpha=0.5, fontsize=8, va='center')

    ax2.set_xlabel("Total Training Energy (J)", fontsize=12)
    ax2.set_ylabel("Information Gain (nats)", fontsize=12)
    ax2.set_title("Experiment 1: Energy Efficiency Comparison", fontsize=14)
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)

    e_all = [p[0] for p in all_points]
    if np.any(np.array(e_all) > 0) and max(e_all)/max(min(e_all), 0.001) > 20:
        ax2.set_xscale('log')
        ax2.set_xlabel("Total Training Energy (J, log scale)", fontsize=12)

    fig2.tight_layout()
    fig2.savefig(output_dir / "exp1_scatter.png", dpi=150)
    print(f"Saved: {output_dir / 'exp1_scatter.png'}")
    plt.close(fig2)

    # ── Figure 3: Energy curves overlay ──────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    for model_name in sorted(all_data.keys()):
        meta = MODEL_META.get(model_name, {"label": model_name, "color": "gray",
                                            "marker": "o", "linestyle": "-"})
        # Use first run's curve
        cf = results_root / model_name / "run_1" / "energy_curve.csv"
        if cf.exists():
            curve = pd.read_csv(cf)
            ax3.plot(curve["cumulative_energy_j"], curve["info_gain_nats"],
                     color=meta["color"], linewidth=1.5,
                     label=meta["label"])
            ax3.scatter(curve["cumulative_energy_j"].iloc[-1],
                        curve["info_gain_nats"].iloc[-1],
                        marker=meta["marker"], color=meta["color"],
                        s=80, edgecolors='white', zorder=5)

    # Threshold lines
    for th, ig in zip(R2_THRESHOLDS, IG_THRESHOLDS):
        ax3.axhline(y=ig, color='red', alpha=0.2, linewidth=0.8)
        ax3.text(ax3.get_xlim()[1]*0.95, ig*1.05, f"R²={th}",
                 color='red', alpha=0.6, fontsize=9, ha='right')

    ax3.set_xlabel("Cumulative Energy (J)", fontsize=12)
    ax3.set_ylabel("Information Gain (nats)", fontsize=12)
    ax3.set_title("Energy-Information Curves (1st run)", fontsize=14)
    ax3.legend(fontsize=9, loc='lower right')
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(output_dir / "exp1_curves_overlay.png", dpi=150)
    print(f"Saved: {output_dir / 'exp1_curves_overlay.png'}")
    plt.close(fig3)

    # ── Figure 4: Single-threshold bar (at R²=0.75) with all models ──
    fig4, ax4 = plt.subplots(figsize=(12, 5))

    col = "e_r2_0.75_mean"
    th = 0.75
    ig_val = -0.5 * np.log(1 - th)
    models_sorted = []
    vals_sorted = []
    colors_sorted = []

    for _, row in df_summary.sort_values(col, ascending=True).iterrows():
        meta = MODEL_META.get(row["model"], {})
        v = row.get(col, np.nan)
        if not np.isnan(v) and v > 0:
            models_sorted.append(meta.get("label", row["model"]))
            vals_sorted.append(v)
            colors_sorted.append(meta.get("color", "gray"))

    bars = ax4.bar(range(len(models_sorted)), vals_sorted, color=colors_sorted,
                   width=0.6, edgecolor='white', linewidth=0.5)
    ax4.set_xticks(range(len(models_sorted)))
    ax4.set_xticklabels(models_sorted, rotation=25, ha='right', fontsize=10)
    ax4.set_ylabel("Energy to Reach R²=0.75 (J)", fontsize=12)
    ax4.set_title(f"Energy Required to Reach R² ≥ 0.75 (IG ≥ {ig_val:.3f} nats)",
                  fontsize=13)
    ax4.grid(True, axis='y', alpha=0.3)

    for bar, v in zip(bars, vals_sorted):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals_sorted)*0.01,
                 f"{v:.1f}J", ha='center', va='bottom', fontsize=9)

    fig4.tight_layout()
    fig4.savefig(output_dir / "exp1_threshold_r2_075.png", dpi=150)
    print(f"Saved: {output_dir / 'exp1_threshold_r2_075.png'}")
    plt.close(fig4)

    print("\nAll outputs:")
    print(f"  {output_dir/'exp1_summary.csv'}")
    print(f"  {output_dir/'exp1_detail.csv'}")
    print(f"  {output_dir/'exp1_threshold_energy.png'}  (3-panel bar)")
    print(f"  {output_dir/'exp1_threshold_r2_075.png'}  (single bar)")
    print(f"  {output_dir/'exp1_scatter.png'}")
    print(f"  {output_dir/'exp1_curves_overlay.png'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
