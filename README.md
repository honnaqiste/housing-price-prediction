# 加州房价预测系统

California Housing Price Prediction | A01 课程选题

基于 1990 年加州人口普查数据的机器学习预测系统。采用随机森林算法实现房价预测（R² ≈ 0.817），提供 Streamlit 交互式估价界面、地理热力图可视化和能耗监测功能。

**技术栈**：Python 3.11, Scikit-learn, Streamlit, Pandas, Matplotlib, Seaborn

---

## 快速开始

### 环境激活与依赖安装

```bash
# 激活 Conda 环境
conda activate python-for-ai

# 安装项目依赖
pip install -r requirements.txt
```

若环境不存在，请先创建：
```bash
conda create -n python-for-ai python=3.11
conda activate python-for-ai
pip install -r requirements.txt
```

### 启动 Streamlit Web 应用

```bash
streamlit run src/app.py
```

应用将在 `http://localhost:8501` 启动，无需额外配置。

---

## 项目概览

### 核心目标

本项目通过对加州住房数据集的深入分析，建立房价预测模型并开发交互式估价工具。数据集包含 20,640 条样本，特征维度为 8 维（地理位置、房龄、房间数等），是机器学习回归问题的经典基准数据集。

### 技术特点

采用多算法对比框架，其中随机森林算法通过非线性特征学习达到最优性能。项目同时集成了能耗监测工具（EnergyTracker），在 Linux RAPL 系统上进行精确的功率测量，在其他平台上优雅降级。Streamlit 前端提供中文界面支持，参数说明详细，预测结果实时反馈。

---

## 项目结构

```
housing-price-prediction/
│
├── data/                              原始数据与处理后数据
│   ├── raw/housing.csv               加州房价原始数据集（20,640条）
│   └── processed/                    标准化与编码后的数据
│       ├── housing_processed.csv
│       └── mlp_results/              MLP模型输出
│
├── models/                            已训练的模型存储库
│   ├── random_forest/                随机森林（最佳模型）
│   │   ├── rf_model.joblib          序列化模型
│   │   ├── preprocessor.joblib       数据预处理管道
│   │   └── feature_importances.png   特征重要性可视化
│   ├── linear/                       线性回归
│   ├── mlp/                          多层感知机
│   └── sgd/                          SGD回归
│
├── src/                               源代码（核心业务逻辑）
│   ├── app.py                        Streamlit Web应用
│   ├── compare_models.py             模型对比与评估脚本
│   ├── preprocessing.py              统一的数据预处理管道
│   ├── predict.py                    批量/交互式预测脚本
│   ├── utils.py                      能耗监测工具
│   ├── train_*.py                    各模型训练脚本
│   └── test.py                       单元测试
│
├── notebooks/                         Jupyter交互式分析
│   └── reference/california-housing-prices.ipynb  EDA与可视化
│
└── experiments/                       实验结果与日志记录
    └── mlp/20260515/                 能耗实验数据
```

---

## 使用指南

### Streamlit 交互式应用

**启动应用：**
```bash
streamlit run src/app.py
```

应用提供以下功能模块：

**侧边栏参数输入** — 通过直观的滑块与数值输入框设置房屋特征。各参数均包含上下文帮助提示，说明参数含义与数据范围。支持的参数包括：地理坐标（经纬度）、房屋中位年龄、房间数、人口数、户数、收入中位数及地理位置类型（内陆、近海、近湾等）。

**实时预测** — 点击"开始预测"按钮，后端加载保存的随机森林模型，即时计算目标房价并在绿色成功框中显示结果（精确到美元）。

**参数说明与结果展示** — 页面顶部展示详细的参数释义（中英文对照），底部展示特征重要性排序图表，帮助用户理解各特征对房价的影响权重。

**技术实现** — 应用采用 Streamlit 框架，集成 scikit-learn 预训练模型与自定义预处理管道，确保生产环境的数据一致性。前端与后端通过 Streamlit 的反应式编程模型解耦，无需额外的 Web 服务器配置。

### 模型训练与对比

```bash
python src/compare_models.py
```

此脚本自动加载数据、训练多个回归模型（线性回归、随机森林、MLP、SGD）、计算评估指标并生成特征重要性图表。运行结果保存在 `models/` 目录下对应的子目录中。

### 批量预测

```bash
python src/predict.py
```

支持交互式逐条输入特征进行预测，或通过以下方式处理 CSV 文件：
```bash
python src/predict.py --input data.csv --output predictions.csv
```

### 数据预处理

如需重新处理原始数据，运行对应的 Jupyter Notebook：
```bash
jupyter notebook data/data_preprocessing.ipynb
```

---

## 模型性能对比

| 模型 | R² 分数 | RMSE（美元）| MAE（美元）| 特点 |
|-----|--------|-----------|----------|------|
| **Random Forest** | **0.817** | **48,975** | 31,500 | 非线性特征学习，表现最佳 |
| Linear Regression | 0.625 | 70,060 | 48,200 | 线性假设，易于解释 |
| MLP | -0.02 | N/A | N/A | 过拟合，需改进超参数 |
| SGD | 0.50 | 95,000 | 65,300 | 收敛困难 |

> **核心发现**：随机森林通过集成 100 棵决策树有效捕捉特征间的非线性关系，在本数据集上表现远优于其他算法。R² = 0.817 意味着模型解释了房价变异的 81.7%。

### 特征重要性分析

| 排序 | 特征 | 重要性分数 | 解释 |
|-----|------|----------|------|
| 1 | Median Income（收入中位数） | 0.568 | 收入是房价最强驱动力 |
| 2 | Longitude（经度） | 0.087 | 地理位置影响显著 |
| 3 | Latitude（纬度） | 0.085 | 南北方向位置差异 |
| 4-10 | 其他特征 | 0.260 | 房龄、人口、房间数等 |

> **关键洞察**：收入中位数单独贡献了 56.8% 的预测权重，表明收入水平是决定房价的最主要因素。地理位置（经纬度）合计贡献 17.2%，反映加州房价的显著地域差异。

---

## 能耗监测 (EnergyTracker)

项目集成了自定义的能耗监测工具 (`src/utils.py`)，用于追踪模型训练的能源消耗。在支持 Intel RAPL 的 Linux 系统上进行精确测量，返回瓦特·秒（J）的能耗数据。在 Windows 及 macOS 等非 RAPL 系统上，工具自动降级，返回 0 值而不报错，确保跨平台兼容性。

EnergyTracker 的核心用法为：启动前调用 `tracker.start()`，训练完成后调用 `tracker.end()`，之后通过 `tracker.save_curve()` 导出能耗曲线。这使得能耗分析与模型优化可同步进行，适合在有功率预算限制的边缘计算场景中使用。

---

## 数据分析发现

### 地理分布特征

房价与地理位置呈高度相关性。加州沿海地区（旧金山湾区、洛杉矶）房价普遍高于内陆地区，最高与最低房价相差 5 倍以上。地理热力图清晰显示房价的空间聚集效应，沿海一线形成明显的高房价带。

### 特征相关性

收入中位数与房价的皮尔逊相关系数为 0.69，是所有单变量特征中最强的。房屋中位年龄与房价呈弱负相关（r ≈ -0.10），表明较新建的房屋平均房价略低。房间数与人口数对房价的直接贡献较小，但通过与其他特征的交互作用间接影响预测。

### 数据质量

数据包含少量缺失值，已通过中位数插补处理。房价分布右偏，部分高端地区房价存在人为上限（$500K）。标准化预处理确保数值特征在 [-1, 1] 范围内，分类特征（Ocean Proximity）采用 One-Hot 编码，共 5 个类别。

详细的探索性分析见 [california-housing-prices.ipynb](notebooks/reference/california-housing-prices.ipynb)。

---

## 常见问题

**Q: Streamlit 应用无法启动？**  
A: 确保已在正确的 Conda 环境中安装 streamlit（`pip list | grep streamlit`）。若仍报错，尝试 `pip install --upgrade streamlit`。

**Q: 预测结果与期望值相差大？**  
A: 检查输入特征是否超出训练数据范围。例如，中位数收入数据范围为 0.5–15 万美元。也可尝试线性回归模型（更保守的预测）。

**Q: 如何修改模型或调整超参数？**  
A: 编辑 `src/compare_models.py` 中的模型初始化代码，修改参数后重新运行以生成新的模型文件。

**Q: 能否部署到生产环境？**  
A: 可。推荐使用 Streamlit Cloud（免费）、Heroku 或 Docker 容器化。详见 [Streamlit 部署文档](https://docs.streamlit.io/deploy)。

---

## 参考资源

- [加州房价数据集 (Kaggle)](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- [Scikit-learn 官方文档](https://scikit-learn.org/)
- [Streamlit 文档](https://docs.streamlit.io/)
- [Pandas 用户指南](https://pandas.pydata.org/docs/)

---

**项目维护者**：[di_7_zu]  
**最后更新**：2026-06-04
