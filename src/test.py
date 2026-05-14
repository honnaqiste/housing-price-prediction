import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

df = pd.read_csv("data/processed/housing_encoded.csv", header=0)

# 删除派生特征
derived = ['rooms_per_household', 'bedrooms_per_room', 'population_per_household']
df = df.drop(columns=[c for c in derived if c in df.columns])

# 目标列
target = 'median_house_value'
y = df[target]
X = df.drop(columns=[target])

# 检查目标分布
print("Target stats:")
print(y.describe())

# 检查特征与目标的相关性
correlations = X.corrwith(y).sort_values(ascending=False)
print("\nTop 5 correlations with target:")
print(correlations.head(5))

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

r2 = r2_score(y_val, y_pred)
print(f"\nR²: {r2:.4f}")

# 检查预测值与真实值的散点图（可选）
import matplotlib.pyplot as plt
plt.scatter(y_val, y_pred, alpha=0.3)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title(f"R²={r2:.3f}")
plt.show()