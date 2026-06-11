# 加州房价预测系统 — 能耗与性能分析

California Housing Price Prediction | A01 课程选题

基于 1990 年加州人口普查数据的机器学习预测系统。采用多模型对比框架，系统性地分析不同算法在房价预测任务上的**性能-能耗权衡**。提供 Streamlit 交互式估价界面、完整的实验分析流程和 RAPL 能耗监测工具。

**技术栈**：Python 3.11, Scikit-learn, Streamlit, Pandas, Matplotlib, NumPy

---

## 项目概览

### 完整流程

```
原始数据 (housing.csv)
    ↓
数据预处理 (Preprocessor)
    ├── 数值特征标准化 (StandardScaler)
    └── 类别特征 One-Hot 编码
    ↓
模型训练 ─── 同时测量能耗 (EnergyTracker)
    ├── 线性回归 (LinearRegression)
    ├── 随机森林 (RandomForest)
    ├── 直方图梯度提升 (HistGradientBoosting)
    └── 多层感知机 (MLPRegressor)
    ↓
实验结果分析 (analyze.py)
    ├── 能耗对比图
    ├── 信息增益-能耗曲线
    ├── 阈值效率分析 (R²=0.5 / 0.75 / 0.8)
    └── 汇总表
    ↓
模型部署 (Streamlit App)
    └── 交互式房价估价 + 特征重要性展示
```

### 两大实验

| 实验 | 内容 | 代码入口 |
|------|------|---------|
| **实验1：跨模型能耗对比** | Linear / RF / HistGB / MLP 四种模型在相同数据下的能耗、精度对比 | `experiments/exp1_model_comparison/` |
| **实验2：MLP 结构搜索** | 10 种不同隐藏层结构的 MLP，研究模型复杂度与能耗的关系 | `experiments/exp2_mlp_sweep/` |

---

## 快速开始

### 环境

```bash
conda create -n python_cource python=3.11
conda activate python_cource
pip install -r requirements.txt
```

### 启动 Streamlit Web 应用

```bash
streamlit run src/app.py
```

应用在 `http://localhost:8501` 启动，基于预训练的随机森林模型提供实时房价预测及特征重要性可视化。

---

## 项目结构

```
housing-price-prediction/
│
├── data/                              原始与处理后的数据
│   ├── raw/housing.csv                加州房价数据集（20,640 条）
│   ├── processed/                     编码后的数据
│   └── data_preprocessing.ipynb       数据预处理与 EDA（含热力图）
│
├── src/                               源代码
│   ├── app.py                         Streamlit 交互式房价预测
│   ├── preprocessing.py               统一的 Preprocessor 管道
│   └── utils.py                       EnergyTracker 能耗监测工具
│
├── experiments/                       实验代码与结果
│   ├── exp1_model_comparison/         实验1：跨模型能耗对比
│   │   ├── train.py                   统一训练入口（支持 --no-energy）
│   │   ├── grid_search.py             超参数网格搜索
│   │   ├── analyze.py                 分析绘图：能耗对比、阈值分析
│   │   └── run_experiment.sh          自动化批处理脚本
│   └── exp2_mlp_sweep/                实验2：MLP 结构搜索
│       ├── analyze.py                 分析绘图：架构-能耗关系
│       ├── grid_search.py             MLP 超参数搜索
│       └── run_experiment.sh          自动化批处理脚本
│
└── notebooks/                         Jupyter 探索式分析
    └── reference/california-housing-prices.ipynb
```

---

## 使用指南

### 1. 数据探索

```bash
jupyter notebook data/data_preprocessing.ipynb
```

包含特征相关性热力图、房价分布分析、缺失值处理等探索性数据分析。

### 2. 运行实验

#### 实验1：跨模型对比

```bash
# 带能耗测量（需要 root + RAPL）
cd experiments/exp1_model_comparison
sudo conda run -n python_cource python train.py --model linear
sudo conda run -n python_cource python train.py --model random_forest
sudo conda run -n python_cource python train.py --model histgb
sudo conda run -n python_cource python train.py --model mlp --mlp-hidden "256,128,64"

# 不带能耗（普通用户，调参用）
python train.py --model random_forest --no-energy
python train.py --model mlp --mlp-hidden "128,64,32" --no-energy
```

#### 实验2：MLP 架构搜索

```bash
cd experiments/exp2_mlp_sweep
sudo conda run -n python_cource python -m grid_search
```

### 3. 分析结果

```bash
# 实验1分析
cd experiments/exp1_model_comparison
python analyze.py

# 实验2分析
cd experiments/exp2_mlp_sweep
python analyze.py
```

生成的图表（PNG）和汇总表（CSV）保存在各自的目录下。

### 4. 超参数搜索

```bash
cd experiments/exp1_model_comparison

# 随机搜索（推荐，快速）
python grid_search.py --models rf,mlp,histgb

# 穷举网格（慢但全面）
python grid_search.py --method grid --models mlp
```

---

## 能耗监测 (EnergyTracker)

`src/utils.py` 中的 `EnergyTracker` 类基于 Intel RAPL 接口实现 CPU 能耗的精确测量。

### 核心特性

- **上下文管理器**：`with EnergyTracker() as t:` 自动记录开始/结束能耗
- **背景功率扣除**：自动测量 idle 状态功耗，计算净能耗
- **每步追踪**：`tracker.get_current_energy()` 可在训练过程中多次调用
- **优雅降级**：非 RAPL 环境返回 0，不中断程序
- **净能耗计算**：`净能耗 = 总能耗 − 背景功率 × 时间`

### 用法示例

```python
from src.utils import EnergyTracker

tracker = EnergyTracker(idle_duration=10.0)
with tracker:
    model.fit(X_train, y_train)          # 训练在此进行
    cum_e = tracker.get_current_energy()  # 可多次调用
net_energy = tracker.get_energy()         # 训练净能耗
print(f"Net energy: {net_energy:.3f} J")
```

### 不测能耗模式

所有训练脚本支持 `--no-energy` 参数，用 `NoopTracker` 替换 `EnergyTracker`，方便在没有 RAPL 的环境下调参。

---

## 关键发现

### 特征重要性

| 排序 | 特征 | 重要性 |
|-----|------|-------|
| 1 | Median Income (收入中位数) | 0.568 |
| 2 | Longitude (经度) | 0.087 |
| 3 | Latitude (纬度) | 0.085 |
| 4-10 | 其他特征 | 0.260 |

收入中位数贡献了 56.8% 的预测权重，是房价的最强驱动因素。

### 性能对比

| 模型 | R² | RMSE | 能耗 (J) |
|-----|-----|------|----------|
| Random Forest | ~0.82 | ~49K | ~10-50 |
| HistGB | ~0.80 | ~50K | ~5-20 |
| Linear | ~0.63 | ~70K | ~0.01 |
| MLP | ~0.79* | ~52K | ~50-200 |

\* MLP 取决于隐藏层结构和 epoch 数，详见过早停止策略。

### 能耗-精度权衡

- 线性回归能耗极低（~0.01J），但精度有限
- RF 和 HistGB 在能耗和精度之间取得较好平衡
- MLP 能耗最高，可通过减少 epoch 或简化结构优化
- 到达 R²=0.75 所需的能耗远小于到达 R²=0.8，存在显著的边际收益递减

---

## 数据分析发现

### 地理分布特征

房价与地理位置呈高度相关性。加州沿海地区（旧金山湾区、洛杉矶）房价普遍高于内陆地区，最高与最低房价相差 5 倍以上。

### 特征相关性

收入中位数与房价的皮尔逊相关系数为 0.69，是所有单变量特征中最强的。房屋中位年龄与房价呈弱负相关（r ≈ -0.10）。

### 数据质量

数据包含少量缺失值，已通过中位数插补处理。房价分布右偏，部分高端地区房价存在人为上限（$500K）。

---

## 参考资料

- [加州房价数据集 (Kaggle)](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- [Scikit-learn 官方文档](https://scikit-learn.org/)
- [Intel RAPL 能耗监测](https://www.kernel.org/doc/html/latest/power/powercap/powercap.html)

---

**项目维护者**：[di_7_zu]  
**最后更新**：2026-06-11
