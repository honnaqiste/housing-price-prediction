#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超参数搜索脚本 — 对 Experiment 1 中测能量的三个模型
（Random Forest、HistGradientBoosting、MLP）搜索最优超参数。
不记录能耗。

默认使用 RandomizedSearchCV（随机搜索，快得多），
也可选 GridSearchCV（穷举网格，慢）。

数据加载方式与 experiments/exp1_model_comparison/train.py 保持一致。

用法:
    python src/grid_search.py                                   # 三个模型，随机搜索
    python src/grid_search.py --method grid                     # 穷举网格（慢）
    python src/grid_search.py --n-iter 20                       # 每个模型只试 20 种组合
    python src/grid_search.py --models rf,mlp                   # 只搜 RF 和 MLP
    python src/grid_search.py --models histgb                   # 只搜 HistGB
    python src/grid_search.py --cv 5                            # 5 折 CV
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, train_test_split
)
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings("ignore", category=FutureWarning)

# 添加项目根目录到 sys.path，让 preprocessing 模块可导入
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import Preprocessor


# ──────────────────────────────────────────────
# 数据加载（与 exp1/train.py 保持一致）
# ──────────────────────────────────────────────
def load_data(data_path, test_size=0.2, random_state=42, log_target=False):
    """
    加载原始 housing.csv。
    先拆 train / val，再在 train 上 fit Preprocessor（只做 X 标准化），
    避免数据泄露。

    如果 log_target=True，对 y 取 np.log(房价)（预测时 exp 还原）。
    否则 y 保持原始房价（美元）。
    —— 不对 y 做标准化，树模型不需要，MLP 也能直接处理。

    返回 (X_train, X_val, y_train, y_val, pre, log_target)。
    """
    df = pd.read_csv(data_path)

    target_col = 'median_house_value'
    y = df[target_col].values.ravel()
    X_df = df.drop(columns=[target_col])

    if log_target:
        print(f'  对目标值取 log 变换 (原范围: ${y.min():.0f} ~ ${y.max():.0f})')
        y = np.log(y)
    else:
        print(f'  目标值保持原始房价 (${y.min():.0f} ~ ${y.max():.0f})')

    # 1) 先划分，再预处理 —— 防止数据泄露
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state
    )

    # 2) Preprocessor 只 fit 训练集（内部有 StandardScaler 标准化 X）
    pre = Preprocessor()
    X_train = pre.fit_transform(X_train_raw)
    X_val = pre.transform(X_val_raw)
    print(f'  特征矩阵 (X 已标准化): {X_train.shape}')
    print(f'  训练集: {X_train.shape[0]:>5}, 验证集: {X_val.shape[0]:>5}  (共计 {X_train.shape[0]+X_val.shape[0]})')
    return X_train, X_val, y_train, y_val, pre, log_target


# ──────────────────────────────────────────────
# 各模型的网格参数定义
# ──────────────────────────────────────────────
def get_rf_param_grid():
    """RandomForestRegressor 超参数搜索空间"""
    return {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 15, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
    }


def get_histgb_param_grid():
    """HistGradientBoostingRegressor 超参数搜索空间"""
    return {
        'max_iter': [100, 200, 500, 1000],
        'max_depth': [None, 3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'min_samples_leaf': [20, 50, 100, 200],
        'l2_regularization': [0.0, 0.1, 0.5, 1.0],
        'max_bins': [255, 128],
    }


def get_mlp_param_grid():
    """MLPRegressor 超参数搜索空间（3 层隐藏层）"""
    return {
        'hidden_layer_sizes': [
            (128, 64, 32),
            (256, 128, 64),
        ],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.01, 0.005, 0.001],
        'learning_rate_init': [0.0005, 0.001],
        'max_iter': [500, 1000],
        'early_stopping': [True],
        'validation_fraction': [0.1],
        'batch_size': [32 , 64, 128],
    }


# ──────────────────────────────────────────────
# 模型名称 → (模型类, 参数网格, 额外关键字)
# ──────────────────────────────────────────────
MODEL_REGISTRY = {
    'rf': {
        'class': RandomForestRegressor,
        'param_grid': get_rf_param_grid,
        'fixed_kwargs': {'random_state': 42, 'n_jobs': -1},
        'label': 'Random Forest',
    },
    'histgb': {
        'class': HistGradientBoostingRegressor,
        'param_grid': get_histgb_param_grid,
        'fixed_kwargs': {'random_state': 42, 'verbose': 0},
        'label': 'HistGradientBoosting',
    },
    'mlp': {
        'class': MLPRegressor,
        'param_grid': get_mlp_param_grid,
        'fixed_kwargs': {'random_state': 42, 'verbose': False},
        'label': 'MLP',
    },
}


# ──────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=('网格搜索：对 Experiment 1 的三个模型'
                     '（Random Forest / HistGB / MLP）搜索最优参数')
    )
    parser.add_argument('--data', default='data/raw/housing.csv',
                        help='数据文件路径（相对于项目根目录，默认 data/raw/housing.csv）')
    parser.add_argument('--models', type=str, default='mlp',
                        help='要搜索的模型，逗号分隔，可选: rf, histgb, mlp')
    parser.add_argument('--cv', type=int, default=3,
                        help='交叉验证折数（默认 3）')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='留出验证集比例（默认 0.2）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--output-dir', type=str,
                        default='experiments/exp1_model_comparison',
                        help='结果输出目录（默认 experiments/exp1_model_comparison/）')
    parser.add_argument('--method', type=str, default='random',
                        choices=['random', 'grid'],
                        help='搜索方法: random=随机搜索(快), grid=穷举网格(慢)')
    parser.add_argument('--raw-target', action='store_true',
                        help='不对房价取 log，直接预测原始房价（默认使用 log 变换改善偏态）')
    parser.add_argument('--n-iter', type=int, default=60,
                        help='随机搜索时每个模型采样的组合数（默认 90）')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='GridSearchCV 并行任务数（默认 -1 = 全部 CPU）')
    args = parser.parse_args()

    data_path = PROJECT_ROOT / args.data
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_models = [m.strip() for m in args.models.split(',')]
    for m in selected_models:
        if m not in MODEL_REGISTRY:
            print(f'❌ 未知模型: "{m}"，可选: {list(MODEL_REGISTRY.keys())}')
            sys.exit(1)

    print(f'📂 加载数据: {data_path}')
    if not data_path.exists():
        print(f'❌ 数据文件不存在: {data_path}')
        sys.exit(1)

    use_log = not args.raw_target
    X_train, X_val, y_train, y_val, pre, log_target = load_data(
        data_path, test_size=args.test_size, random_state=args.seed,
        log_target=use_log
    )

    if log_target:
        print(f'   y 是 log(房价)，验证时 exp 还原到美元')
        print(f'   训练集 log(房价) 范围: [{y_train.min():.4f}, {y_train.max():.4f}]')
    else:
        print(f'   y 是原始房价（美元）')
        print(f'   训练集房价范围: [${y_train.min():,.0f}, ${y_train.max():,.0f}]')

    method_name = 'RandomizedSearchCV (随机搜索)' if args.method == 'random' else 'GridSearchCV (穷举网格)'
    print(f'🔍 搜索模型: {selected_models}')
    print(f'🏁 搜索方法: {method_name}')
    print(f'🏁 CV 折数: {args.cv}')
    if args.method == 'random':
        print(f'🏁 每模型采样: {args.n_iter} 种组合（总组合数远大于此）')
    print('=' * 70)

    all_results = {}

    for model_name in selected_models:
        print(f'\n{"─" * 70}')
        entry = MODEL_REGISTRY[model_name]
        label = entry['label']
        print(f'🚀 开始搜索: {label} ({model_name})')
        print(f'{"─" * 70}')

        model_class = entry['class']
        param_grid = entry['param_grid']()
        fixed_kwargs = entry['fixed_kwargs']

        # 估算总组合数
        total_combos = 1
        for v in param_grid.values():
            total_combos *= len(v)
        n_train = total_combos if args.method == 'grid' else args.n_iter
        print(f'   参数: {len(param_grid)} 维, 总组合数: {total_combos}')
        print(f'   实际训练: {n_train} × {args.cv} = {n_train * args.cv} 个模型')

        # 创建模型实例
        model = model_class(**fixed_kwargs)

        # 选择搜索器
        if args.method == 'random':
            searcher = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=args.n_iter,
                cv=args.cv,
                scoring='neg_mean_squared_error',
                n_jobs=args.n_jobs,
                verbose=1,
                return_train_score=True,
                random_state=args.seed,
            )
        else:
            searcher = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=args.cv,
                scoring='neg_mean_squared_error',
                n_jobs=args.n_jobs,
                verbose=1,
                return_train_score=True,
            )

        print(f'   训练中...')
        searcher.fit(X_train, y_train)
        gs = searcher

        # ---- 最佳模型在留出的验证集上评估（原始美元尺度） ----
        best_estimator = gs.best_estimator_
        y_val_pred = best_estimator.predict(X_val)

        # 如果是对数空间，exp 回到美元用于计算指标
        if log_target:
            y_val_pred_dollar = np.exp(y_val_pred)
            y_val_dollar = np.exp(y_val)
        else:
            y_val_pred_dollar = y_val_pred
            y_val_dollar = y_val

        val_rmse = float(np.sqrt(mean_squared_error(y_val_dollar, y_val_pred_dollar)))
        val_mae = float(mean_absolute_error(y_val_dollar, y_val_pred_dollar))
        val_r2 = float(r2_score(y_val_dollar, y_val_pred_dollar))

        # CV 的 MSE（直接就是 y 原始空间的值，因为不再对 y 做标准化）
        mse_cv = float(-gs.best_score_)

        result = {
            'model': model_name,
            'best_params': gs.best_params_,
            'best_score_cv_mse': mse_cv,
            'best_rank': int(gs.best_index_),
            'validation': {
                'RMSE': val_rmse,
                'MAE': val_mae,
                'R2': val_r2,
            },
            'n_cv_folds': args.cv,
            'log_target': log_target,
        }

        all_results[model_name] = result

        # 打印结果
        print(f'\n✅ {label} 搜索完成！')
        print(f'   最佳参数:')
        for k, v in gs.best_params_.items():
            print(f'      {k} = {v}')
        print(f'   CV RMSE:      {mse_cv ** 0.5:,.0f} 美元')
        print(f'   验证集 RMSE:  {val_rmse:,.2f} 美元')
        print(f'   验证集 MAE:   {val_mae:,.2f} 美元')
        print(f'   验证集 R²:    {val_r2:.4f}')

        # Top-5 参数组合（按 CV RMSE 排序，单位与 y 一致）
        cv_results = pd.DataFrame(gs.cv_results_)
        top5_cols = [c for c in cv_results.columns
                     if c.startswith('param_') or c in (
                         'mean_test_score', 'rank_test_score', 'std_test_score')]
        top5 = (cv_results.sort_values('rank_test_score')
                          .head(5)[top5_cols].copy())
        top5['rmse'] = np.sqrt(-top5['mean_test_score'])
        unit = 'log($)' if log_target else '$'
        print(f'\n   📋 Top-5 参数组合 (CV RMSE, 单位={unit}):')
        for i, (_, row) in enumerate(top5.iterrows()):
            params = {k.replace('param_', ''): v
                      for k, v in row.items()
                      if k.startswith('param_')}
            params_str = ', '.join(f'{k}={v}' for k, v in params.items())
            print(f'      #{i+1}: RMSE={row["rmse"]:,.2f} | {params_str}')

    # ── 保存全部结果 ──
    results_path = output_dir / 'grid_search_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f'\n{"=" * 70}')
    print(f'💾 全部结果已保存至: {results_path}')

    # ── 汇总表 ──
    print(f'\n{"=" * 70}')
    print(f'📊 汇总对比')
    print(f'{"=" * 70}')
    summary_data = []
    for name, res in all_results.items():
        label = MODEL_REGISTRY[name]['label']
        # 精简打印参数
        params_short = ', '.join(
            f'{k}={v}' for k, v in res['best_params'].items()
        )
        cv_rmse = res['best_score_cv_mse'] ** 0.5
        unit = 'log$' if res.get('log_target') else '$'
        summary_data.append({
            '模型': label,
            '最佳参数摘要': params_short[:100],
            f'CV RMSE({unit})': f'{cv_rmse:,.2f}',
            '验证 RMSE($)': f'{res["validation"]["RMSE"]:,.0f}',
            '验证 R²': f'{res["validation"]["R2"]:.4f}',
        })
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # 保存汇总 CSV
    summary_csv = output_dir / 'grid_search_summary.csv'
    summary_df.to_csv(summary_csv, index=False, encoding='utf-8')
    print(f'\n💾 汇总表已保存至: {summary_csv}')

    # ── 简单对比图 ──
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        names = []
        r2_vals = []
        rmse_vals = []
        for name, res in all_results.items():
            names.append(MODEL_REGISTRY[name]['label'])
            r2_vals.append(res['validation']['R2'])
            rmse_vals.append(res['validation']['RMSE'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        colors = ['#4CAF50', '#FF9800', '#E91E63']
        bars1 = ax1.bar(names, r2_vals, color=colors[:len(names)])
        ax1.set_ylabel('R² Score')
        ax1.set_title('验证集 R² 对比')
        ax1.set_ylim(0, 1)
        for bar, v in zip(bars1, r2_vals):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{v:.4f}', ha='center', va='bottom', fontsize=10)

        bars2 = ax2.bar(names, rmse_vals, color=colors[:len(names)])
        ax2.set_ylabel('RMSE (美元)')
        ax2.set_title('验证集 RMSE 对比')
        for bar, v in zip(bars2, rmse_vals):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                     f'${v:,.0f}', ha='center', va='bottom', fontsize=9)

        fig.suptitle('网格搜索结果对比', fontsize=14, fontweight='bold')
        fig.tight_layout()
        plot_path = output_dir / 'grid_search_comparison.png'
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f'💾 对比图已保存至: {plot_path}')
    except Exception as e:
        print(f'⚠️  对比图生成跳过: {e}')

    print(f'\n✅ 网格搜索全部完成！')


if __name__ == '__main__':
    main()
