#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传统 MLP 回归训练脚本（无能耗测量）
数据读取方式与能耗脚本一致（使用项目根目录相对路径）
数据划分为训练集、验证集、测试集（60% / 20% / 20%）
支持早停，最终输出三个集合上的评估指标
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from fontTools.misc.iterTools import batched
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer


def main():
    parser = argparse.ArgumentParser(description="传统 MLP 回归训练（无能耗监测）")
    parser.add_argument("--data", default="data/processed/housing_encoded.csv",
                        help="数据文件路径（相对于项目根目录）")
    parser.add_argument("--hidden-layers", type=str, default="50,50,50",
                        help="隐藏层结构，逗号分隔，例如 '100,50'")
    parser.add_argument("--max-epochs", type=int, default=500,
                        help="最大训练轮数（epoch）")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="初始学习率")
    parser.add_argument("--patience", type=int, default=20,
                        help="早停耐心值（验证集MSE连续多少次不下降则停止）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="测试集比例（剩余数据中验证集占25%，总体训练:验证:测试=60:20:20）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录（默认与数据文件同目录下的 mlp_results）")
    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)

    # 构建数据路径（项目根目录 = 当前脚本所在目录的父目录）
    script_dir = Path(__file__).parent
    proj_root = script_dir.parent
    data_path = proj_root / args.data
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")

    # 确定输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_path.parent / "mlp_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取数据: {data_path}")
    df = pd.read_csv(data_path)

    # 目标列（通常为 'median_house_value'）
    target = 'median_house_value' if 'median_house_value' in df.columns else df.columns[-1]
    print(f"目标列: {target}")

    X = df.drop(columns=[target])
    y = df[target]

    # 区分数值列和类别列
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # 预处理流水线
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )

    # 1. 先划分训练+验证 与 测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )
    # 2. 从临时集中再划分训练集和验证集（验证集占剩余数据的 25%，即总体 20%）
    val_ratio = args.test_size / (1 - args.test_size)  # 0.2/0.8 = 0.25
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=args.seed
    )

    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"测试集大小: {len(X_test)}")

    # 预处理（拟合训练集，转换验证集和测试集）
    X_train_pre = preprocessor.fit_transform(X_train)
    X_val_pre = preprocessor.transform(X_val)
    X_test_pre = preprocessor.transform(X_test)

    # 目标值标准化（MLP通常建议标准化）
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_val_scaled = y_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
    y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

    # 解析隐藏层
    hidden_layers = tuple(map(int, args.hidden_layers.split(',')))
    print(f"MLP 结构: {hidden_layers}")

    # 创建 MLP 模型（逐 epoch 训练，手动早停）
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        learning_rate_init=args.learning_rate,
        max_iter=1,          # 每次partial_fit只训练一个epoch
        warm_start=True,
        random_state=args.seed,
        verbose=False,
        batch_size='auto'
    )

    # 手动早停训练
    best_val_mse = np.inf
    no_improve = 0
    best_epoch = 0
    best_model_coefs = None
    best_model_intercepts = None

    print("开始训练...")
    for epoch in range(1, args.max_epochs + 1):
        model.partial_fit(X_train_pre, y_train_scaled)
        y_val_pred_scaled = model.predict(X_val_pre)
        val_mse = mean_squared_error(y_val_scaled, y_val_pred_scaled)

        if val_mse < best_val_mse - 1e-6:
            best_val_mse = val_mse
            no_improve = 0
            best_epoch = epoch
            # 保存最佳模型参数（coefs_ 和 intercepts_ 是列表，需深拷贝）
            best_model_coefs = [coef.copy() for coef in model.coefs_]
            best_model_intercepts = [intercept.copy() for intercept in model.intercepts_]
        else:
            no_improve += 1

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: val MSE = {val_mse:.6f} (best: {best_val_mse:.6f})")

        if no_improve >= args.patience:
            print(f"早停于 epoch {epoch}，最佳 epoch = {best_epoch}")
            break

    # 恢复最佳模型参数
    if best_model_coefs is not None:
        model.coefs_ = best_model_coefs
        model.intercepts_ = best_model_intercepts
        model.n_iter_ = best_epoch
        print("已恢复最佳模型参数")

    # 在三个集合上评估（原始尺度）
    def evaluate(X_data, y_true, name):
        y_pred_scaled = model.predict(X_data)
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"\n========== {name} 集 ==========")
        print(f"MAE:  {mae:.2f} 美元")
        print(f"RMSE: {rmse:.2f} 美元")
        print(f"R²:   {r2:.4f}")
        return mae, rmse, r2

    print("\n最终评估：")
    evaluate(X_train_pre, y_train, "训练")
    evaluate(X_val_pre, y_val, "验证")
    evaluate(X_test_pre, y_test, "测试")

    # 可选：保存模型和预处理器
    import joblib
    model_path = output_dir / "mlp_model.joblib"
    preprocessor_path = output_dir / "preprocessor.joblib"
    y_scaler_path = output_dir / "y_scaler.joblib"
    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(y_scaler, y_scaler_path)
    print(f"\n模型已保存至: {model_path}")
    print(f"预处理器已保存至: {preprocessor_path}")
    print(f"目标值缩放器已保存至: {y_scaler_path}")


if __name__ == "__main__":
    main()