#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare Linear Regression, Random Forest and MLP on the housing dataset.
Outputs RMSE, MAE, R2 and saves models and feature-importance plot for RF.
"""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from preprocessing import Preprocessor


def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': float(rmse), 'MAE': float(mae), 'R2': float(r2)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed/housing_encoded.csv')
    parser.add_argument('--target', default='median_house_value')
    parser.add_argument('--output', default='models')
    args = parser.parse_args()

    proj_root = Path(__file__).parent
    data_path = proj_root / args.data
    out_dir = proj_root / args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        args.target = df.columns[-1]
    y = df[args.target].values
    X = df.drop(columns=[args.target])

    # Preprocess
    pre = Preprocessor()
    X_proc = pre.fit_transform(X)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42)

    results = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results['linear'] = metrics(y_test, y_pred_lr)
    # save
    lr_dir = out_dir / 'linear'
    lr_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, lr_dir / 'linear_model.joblib')
    joblib.dump(pre, lr_dir / 'preprocessor.joblib')

    # Random Forest with GridSearchCV (simple grid)
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20]
    }
    gs = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    gs.fit(X_train, y_train)
    rf_best = gs.best_estimator_
    y_pred_rf = rf_best.predict(X_test)
    results['random_forest'] = metrics(y_test, y_pred_rf)
    rf_dir = out_dir / 'random_forest'
    rf_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_best, rf_dir / 'rf_model.joblib')
    joblib.dump(pre, rf_dir / 'preprocessor.joblib')
    # save grid search cv results
    with open(rf_dir / 'gridsearch_best_params.json', 'w') as f:
        json.dump({'best_params': gs.best_params_}, f, indent=2)

    # Feature importances plot (map to feature names)
    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = [f'f{i}' for i in range(X_proc.shape[1])]

    importances = rf_best.feature_importances_
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title('Random Forest Feature Importances')
    plt.bar(range(len(importances)), importances[idx], align='center')
    plt.xticks(range(len(importances)), [feat_names[i] for i in idx], rotation=90)
    plt.tight_layout()
    plt.savefig(rf_dir / 'feature_importances.png', dpi=150)
    plt.close()

    # MLP
    mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred_mlp = mlp.predict(X_test)
    results['mlp'] = metrics(y_test, y_pred_mlp)
    mlp_dir = out_dir / 'mlp'
    mlp_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(mlp, mlp_dir / 'mlp_model.joblib')
    joblib.dump(pre, mlp_dir / 'preprocessor.joblib')

    # Save results table
    results_df = pd.DataFrame(results).T
    results_df.to_csv(out_dir / 'model_comparison.csv')
    print(results_df)


if __name__ == '__main__':
    main()
