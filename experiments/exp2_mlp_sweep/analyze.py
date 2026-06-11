#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 2: MLP Architecture Comparison — Analysis.
Thresholds: R² = 0.5, 0.75, 0.799
Output: summary, bar charts, energy curves overlay.
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# R² thresholds
R2_THRESHOLDS = [0.5, 0.75, 0.8]
IG_THRESHOLDS = [-0.5 * np.log(max(1 - r, 1e-10)) for r in R2_THRESHOLDS]

MODEL_META = {
    "mlp_250x30":      {"label": "MLP (250,30)",      "color": "#E91E63", "marker": "D"},
    "mlp_128x64x32":   {"label": "MLP (128,64,32)",   "color": "#4CAF50", "marker": "s"},
    "mlp_32x64x128":   {"label": "MLP (32,64,128)",   "color": "#9C27B0", "marker": "v"},
    "mlp_70x70x70":    {"label": "MLP (70,70,70)",    "color": "#FF9800", "marker": "p"},
    "mlp_90x60x50x30": {"label": "MLP (90,60,50,30)", "color": "#00BCD4", "marker": "^"},
    "mlp_40x150x40":   {"label": "MLP (40,150,40)",   "color": "#FF5722", "marker": "h"},
}


def find_first_above(curve_df, col, threshold):
    mask = curve_df[col] >= threshold
    if mask.any():
        return curve_df.loc[mask.idxmax(), "cumulative_energy_j"]
    return np.nan


def find_first_step(curve_df, col, threshold):
    mask = curve_df[col] >= threshold
    if mask.any():
        return int(curve_df.loc[mask.idxmax(), "step"])
    return np.nan


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

    # ── Scan ─────────────────────────────────────────────────
    rows = []
    detail = []
    all_runs = {}

    for tag_dir in sorted(results_root.iterdir()):
        if not tag_dir.is_dir():
            continue
        tag = tag_dir.name
        runs = []
        for run_dir in sorted(tag_dir.glob("run_*/")):
            mf = run_dir / "metrics.json"
            cf = run_dir / "energy_curve.csv"
            if not mf.exists():
                continue
            with open(mf) as f:
                m = json.load(f)
            m["_run"] = run_dir.name
            runs.append(m)

            dr = {"model": tag, "run": run_dir.name, "seed": m["seed"],
                   "energy_j": m["total_energy_j"], "val_r2": m["final_val_r2"],
                   "info_gain": m["final_info_gain"],
                   "best_epoch": m.get("best_epoch", "")}
            if cf.exists():
                curve = pd.read_csv(cf)
                curve["cumulative_energy_j"] = curve["cumulative_energy_j"].clip(lower=0)
                for th in R2_THRESHOLDS:
                    val = find_first_above(curve, "val_r2", th)
                    dr[f"e_to_r2_{th}"] = val
                    m[f"energy_to_r2_{th}"] = val
                    step_val = find_first_step(curve, "val_r2", th)
                    dr[f"step_to_r2_{th}"] = step_val
                    m[f"step_to_r2_{th}"] = step_val
            # energy per epoch (from curve at best_epoch, not total)
            best_ep = m.get("best_epoch", np.nan)
            energy_at_best = np.nan
            if cf.exists() and not np.isnan(best_ep) and best_ep > 0:
                curve = pd.read_csv(cf)
                row = curve[curve["step"] == best_ep]
                if not row.empty:
                    energy_at_best = row.iloc[0]["cumulative_energy_j"]
                    dr["energy_per_epoch"] = energy_at_best / best_ep
                    dr["energy_at_best"] = energy_at_best
                    m["energy_per_epoch"] = energy_at_best / best_ep
                    m["energy_at_best"] = energy_at_best
            detail.append(dr)

        all_runs[tag] = runs
        if not runs:
            continue
        energies = [r["total_energy_j"] for r in runs]
        gains = [r["final_info_gain"] for r in runs]
        r2s = [r["final_val_r2"] for r in runs]
        row = {"model": tag, "n": len(runs),
               "energy_mean": np.mean(energies), "energy_std": np.std(energies, ddof=1),
               "r2_mean": np.mean(r2s), "gain_mean": np.mean(gains)}
        for th in R2_THRESHOLDS:
            col = f"energy_to_r2_{th}"
            vals = [r.get(col, np.nan) for r in runs]
            vals = [v for v in vals if not np.isnan(v)]
            if not vals:
                row[col] = np.nan
                row[f"{col}_n_reached"] = 0
            else:
                row[col] = np.median(vals) if th >= 0.8 else np.mean(vals)
                row[f"{col}_n_reached"] = len(vals)
        rows.append(row)

    if not rows:
        print("No results found.")
        return

    df = pd.DataFrame(rows)
    df_detail = pd.DataFrame(detail)
    df_detail.to_csv(output_dir / "exp2_detail.csv", index=False)

    print("\n=== Aggregated ===")
    print(df.to_string(index=False))
    df.to_csv(output_dir / "exp2_summary.csv", index=False)

    # ── Figure 1: 3-panel bar ────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, th in enumerate(R2_THRESHOLDS):
        ax = axes[idx]
        col = f"energy_to_r2_{th}"
        labels, vals, colors = [], [], []
        for _, r in df.sort_values(col, ascending=True).iterrows():
            v = r.get(col, np.nan)
            if not np.isnan(v) and v > 0:
                meta = MODEL_META.get(r["model"], {})
                labels.append(meta.get("label", r["model"]))
                vals.append(v)
                colors.append(meta.get("color", "gray"))
        bars = ax.barh(range(len(labels)), vals, color=colors, height=0.6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Energy (J)")
        ig = -0.5 * np.log(max(1 - th, 1e-10))
        suffix = "median" if th >= 0.8 else "mean"
        ax.set_title(f"R² ≥ {th}  ({suffix})", fontsize=11)
        ax.grid(True, axis='x', alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(v + max(vals)*0.01, bar.get_y() + bar.get_height()/2,
                    f"{v:.1f}J", va='center', fontsize=8)
    fig.suptitle("Experiment 2: Energy to Reach R² Thresholds", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "exp2_threshold_energy.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved: {output_dir / 'exp2_threshold_energy.png'}")
    plt.close(fig)

    # ── Figure 2: Single bar at R²=0.8 (best run) ──────────
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    col = "energy_to_r2_0.8"
    labels, vals, colors, notes = [], [], [], []
    for _, r in df.sort_values(col, ascending=True).iterrows():
        v = r.get(col, np.nan)
        if not np.isnan(v) and v > 0:
            meta = MODEL_META.get(r["model"], {})
            labels.append(meta.get("label", r["model"]))
            vals.append(v)
            colors.append(meta.get("color", "gray"))
            n_reached = r.get(f"{col}_n_reached", r["n"])
            notes.append(f"{v:.1f}J (median of {n_reached}/{r['n']} runs)")
    if vals:
        bars = ax2.bar(range(len(labels)), vals, color=colors, width=0.6, edgecolor='white')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
        ax2.set_ylabel("Energy (J)")
        ax2.set_title("Median Energy to Reach R² ≥ 0.80")
        ax2.grid(True, axis='y', alpha=0.3)
        for bar, v, note in zip(bars, vals, notes):
            ax2.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(vals)*0.02,
                     note, ha='center', fontsize=8, rotation=0)
    fig2.tight_layout()
    fig2.savefig(output_dir / "exp2_threshold_r2_08.png", dpi=150)
    print(f"Saved: {output_dir / 'exp2_threshold_r2_08.png'}")
    plt.close(fig2)

    # ── Figure 3: Mean energy curves (avg of 5 runs) ────────
    N_GRID = 300
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    for tag, meta in MODEL_META.items():
        curves = []
        for i in range(1, 6):
            cf = results_root / tag / f"run_{i}" / "energy_curve.csv"
            if cf.exists():
                c = pd.read_csv(cf)
                c["cumulative_energy_j"] = c["cumulative_energy_j"].clip(lower=0)
                curves.append(c)

        if len(curves) < 2:
            continue

        # Common energy grid (truncate to shortest run's max energy)
        min_max_e = min(c["cumulative_energy_j"].max() for c in curves)
        if min_max_e <= 0:
            continue
        energy_grid = np.linspace(0, min_max_e, N_GRID)

        # Interpolate each run onto the grid
        ig_interp = []
        for c in curves:
            e_vals = c["cumulative_energy_j"].values
            ig_vals = c["info_gain_nats"].values
            ig_interp.append(np.interp(energy_grid, e_vals, ig_vals))

        ig_mean = np.mean(ig_interp, axis=0)
        ig_std = np.std(ig_interp, axis=0, ddof=1)

        ax3.plot(energy_grid, ig_mean, color=meta["color"],
                 linewidth=2, label=meta["label"])

    # Threshold lines
    for th, ig in zip(R2_THRESHOLDS, IG_THRESHOLDS):
        ax3.axhline(y=ig, color='red', alpha=0.25, linewidth=0.8)
        ax3.text(ax3.get_xlim()[1]*0.96, ig*1.05, f"R²={th}",
                 color='red', alpha=0.6, fontsize=9, ha='right')

    ax3.set_xlabel("Cumulative Energy (J)", fontsize=12)
    ax3.set_ylabel("Information Gain (nats)", fontsize=12)
    ax3.set_title("Experiment 2: Mean Energy-Information Curves (5 runs)", fontsize=13)
    ax3.legend(fontsize=9, loc='lower right')
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(output_dir / "exp2_curves_overlay.png", dpi=150)
    print(f"Saved: {output_dir / 'exp2_curves_overlay.png'}")
    plt.close(fig3)

    # ── Figure 4: Energy per epoch bar chart ────────────────
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    labels, vals, colors = [], [], []
    for _, r in df.sort_values("energy_mean", ascending=True).iterrows():
        tag = r["model"]
        # compute avg energy per epoch from all runs
        epe_vals = [m.get("energy_per_epoch", np.nan) for m in all_runs.get(tag, [])]
        epe_vals = [v for v in epe_vals if not np.isnan(v)]
        if epe_vals:
            meta = MODEL_META.get(tag, {})
            labels.append(meta.get("label", tag))
            vals.append(np.mean(epe_vals))
            colors.append(meta.get("color", "gray"))
    if vals:
        bars = ax4.bar(range(len(labels)), vals, color=colors, width=0.6, edgecolor='white')
        ax4.set_xticks(range(len(labels)))
        ax4.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
        ax4.set_ylabel("Energy per Epoch (J)")
        ax4.set_title("Average Energy Consumption per Training Epoch")
        ax4.grid(True, axis='y', alpha=0.3)
        for bar, v in zip(bars, vals):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                     f"{v:.3f}J", ha='center', fontsize=9)
    fig4.tight_layout()
    fig4.savefig(output_dir / "exp2_energy_per_epoch.png", dpi=150)
    print(f"Saved: {output_dir / 'exp2_energy_per_epoch.png'}")
    plt.close(fig4)

    # ── Figure 5: Epochs to thresholds ──────────────────────
    EPOCH_THRESHOLDS = [0.5, 0.75, 0.8]
    fig5, ax5 = plt.subplots(figsize=(11, 6))
    x = np.arange(len(MODEL_META))
    width = 0.25
    tags_sorted = sorted(MODEL_META.keys())
    colors_th = ["#4CAF50", "#FF9800", "#E91E63"]

    for idx, th in enumerate(EPOCH_THRESHOLDS):
        agg_func = np.median if th >= 0.8 else np.mean
        suffix = "median" if th >= 0.8 else "mean"
        vals = []
        for tag in tags_sorted:
            runs_data = all_runs.get(tag, [])
            step_vals = [m.get(f"step_to_r2_{th}", np.nan) for m in runs_data]
            step_vals = [v for v in step_vals if not np.isnan(v)]
            vals.append(agg_func(step_vals) if step_vals else 0)

        bars = ax5.bar(x + idx * width, vals, width,
                       label=f"R²≥{th} ({suffix})", color=colors_th[idx],
                       edgecolor='white')
        for bar, v in zip(bars, vals):
            if v > 0:
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f"{v:.0f}", ha='center', fontsize=8)

    labels = [MODEL_META[t]["label"] for t in tags_sorted]
    ax5.set_xticks(x + width)
    ax5.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
    ax5.set_ylabel("Epochs")
    ax5.set_title("Epochs Required to Reach R² Thresholds")
    ax5.legend(fontsize=9)
    ax5.grid(True, axis='y', alpha=0.3)
    fig5.tight_layout()
    fig5.savefig(output_dir / "exp2_epochs_to_threshold.png", dpi=150)
    print(f"Saved: {output_dir / 'exp2_epochs_to_threshold.png'}")
    plt.close(fig5)

    print("\nDone.")


if __name__ == "__main__":
    main()
