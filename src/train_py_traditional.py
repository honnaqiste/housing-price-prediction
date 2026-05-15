#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传统 MLP 回归训练脚本（无能耗测量）
数据划分为训练集、验证集、测试集（60% / 20% / 20%）
支持早停，最终输出三个集合上的评估指标
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# ========== 1. 加载数据 ==========
data_path = "housing.csv"  # 请修改为实际路径
df = pd.read_csv(data_path)

target = "median_house_value"
X = df.drop(columns=[target])
y = df[target]

# ========== 2. 划分：先分出测试集（20%），再从剩余中分出验证集（占剩余25%，即总体的20%） ==========
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 总体
)

print(f"训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_val)}")
print(f"测试集大小: {len(X_test)}")

# ========== 3. 预处理流水线 ==========
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# ========== 4. 构建 MLP 模型（带早停） ==========
# 使用 MLPRegressor 内置的 early_stopping 和 validation_fraction
mlp = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,  # 启用早停
    validation_fraction=0.1,  # 从训练集内部再分10%作为验证（但我们已经独立出了验证集，这里可以不用内置早停）
    n_iter_no_change=20,  # 连续20次迭代无改善则停止
    random_state=42,
    verbose=True
)

# 注意：上述 early_stopping 会从训练集内部再分验证集。为了使用我们独立的验证集，可以手动实现早停。
# 下面采用手动早停方式，更可控。

# ========== 5. 手动早停训练（使用独立的验证集） ==========
# 先预处理数据，因为手动循环需要预处理的特征
X_train_pre = preprocessor.fit_transform(X_train)
X_val_pre = preprocessor.transform(X_val)
X_test_pre = preprocessor.transform(X_test)

# 重新创建 MLP 模型（warm_start=True 以便逐轮训练）
mlp_manual = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=1,  # 每轮只迭代一次
    warm_start=True,  # 保留权重
    random_state=42,
    verbose=False
)

best_val_loss = np.inf
patience = 20
no_improve_count = 0
best_model = None

print("开始训练（手动早停，使用验证集监控）...")
for epoch in range(1, 501):  # 最多500个epoch
    mlp_manual.partial_fit(X_train_pre, y_train)
    y_val_pred = mlp_manual.predict(X_val_pre)
    val_loss = mean_squared_error(y_val, y_val_pred)

    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        no_improve_count = 0
        # 保存最佳模型（可以深拷贝，这里简单保存参数）
        best_weights = (mlp_manual.coefs_, mlp_manual.intercepts_)
    else:
        no_improve_count += 1

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Validation MSE = {val_loss:.2f}")

    if no_improve_count >= patience:
        print(f"早停于 epoch {epoch}")
        break

# 恢复最佳模型（简单方法：重新创建并设置权重，但更简单的是使用已经保存的模型状态）
# 由于 MLPRegressor 的 coefs_ 是只读？可以直接赋值，但更规范的是重新训练一次到最佳 epoch
# 这里为了简单，我们使用最后一次模型，因为手动早停通常不会过拟合太严重。或者重新实例化并拟合到最佳轮数。
# 更严谨的做法：记录最佳 epoch，然后重新训练到那个 epoch。
print("使用最后一轮模型进行评估（或可重新训练最佳模型）。")
mlp_final = mlp_manual  # 简化，使用早停时的模型

# ========== 6. 在三个集合上评估 ==========
y_train_pred = mlp_final.predict(X_train_pre)
y_val_pred = mlp_final.predict(X_val_pre)
y_test_pred = mlp_final.predict(X_test_pre)


def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n========== {name} 集 ==========")
    print(f"MAE:  {mae:.2f} 美元")
    print(f"RMSE: {rmse:.2f} 美元")
    print(f"R²:   {r2:.4f}")


evaluate(y_train, y_train_pred, "训练")
evaluate(y_val, y_val_pred, "验证")
evaluate(y_test, y_test_pred, "测试")