#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
EXPRUN = ROOT_DIR / 'experiments' / 'exp1_model_comparison' / 'results' / 'random_forest' / 'run_1'
MODEL_PATH = EXPRUN / 'model.joblib'
PREPROCESSOR_PATH = EXPRUN / 'preprocessor.joblib'
Y_SCALER_PATH = EXPRUN / 'y_scaler.joblib'
FEATURE_IMPORTANCE_PATH = EXPRUN / 'feature_importances.png'

missing_dependencies = []
if importlib.util.find_spec('sklearn') is None:
    missing_dependencies.append('scikit-learn')

OCEAN_OPTIONS = {
    '内陆地区（INLAND）': 'INLAND',
    '靠近海岸（NEAR OCEAN）': 'NEAR OCEAN',
    '靠近海湾（NEAR BAY）': 'NEAR BAY',
    '1小时车程内达海（<1H OCEAN）': '<1H OCEAN',
    '岛屿地区（ISLAND）': 'ISLAND'
}

st.set_page_config(page_title='加州房价预测', layout='wide')

st.title('加州房价预测')
st.write('使用保存的随机森林模型预测社区/区域级别的中位数房价，并展示重要特征。')

st.subheader('参数说明：')
st.markdown(
    '''
- 地理坐标（Longitude / Latitude）：该区域的东西向经度与南北向纬度  
- 房屋中位年龄（Housing Median Age）：统计区块内房屋建筑年限的中位数（单位：年）  
- 房间总数（Total Rooms）：该区块内所有住宅的房间总数  
- 卧室总数（Total Bedrooms）：该区块内所有住宅的卧室总数  
- 区域人口（Population）：该统计区块内的常住总人口  
- 住户总数（Households）：该区块内的家庭单元（住户）总数  
- 收入中位数（Median Income）：该区域住户的年收入中位数（单位：万美元，原数据为 10,000 为基准）  
- 地理位置类型（Ocean Proximity）：房屋相对于海岸线的地理特征
    ''')

with st.sidebar:
    st.header('请输入房屋特征')
    longitude = st.slider(
        '经度（Longitude）', -125.0, -114.0, -119.5, 0.01,
        help='该区域的东西向经度坐标，数值越大表示越靠东。'
    )
    latitude = st.slider(
        '纬度（Latitude）', 32.0, 42.0, 34.5, 0.01,
        help='该区域的南北向纬度坐标，数值越大表示越靠北。'
    )
    housing_median_age = st.slider(
        '房屋中位年龄（Housing Median Age）', 1, 52, 25, 1,
        help='统计区块内房屋建筑年限的中位数，单位为年。'
    )
    total_rooms = st.number_input(
        '房间总数（Total Rooms）', min_value=2, max_value=10000, value=2000, step=1,
        help='该区块内所有住宅的房间总数，包括卧室、客厅、厨房等。'
    )
    total_bedrooms = st.number_input(
        '卧室总数（Total Bedrooms）', min_value=1, max_value=3000, value=500, step=1,
        help='该区块内所有住宅的卧室总数。'
    )
    population = st.number_input(
        '区域人口（Population）', min_value=1, max_value=20000, value=1200, step=1,
        help='该统计区块内的常住总人口。'
    )
    households = st.number_input(
        '住户总数（Households）', min_value=1, max_value=5000, value=450, step=1,
        help='该区块内的家庭单元（住户）总数。'
    )
    median_income = st.number_input(
        '收入中位数（Median Income，单位：万美元）', min_value=0.5, max_value=15.0, value=3.0, step=0.01, format='%.2f',
        help='该区域住户的年收入中位数，单位为万美元。'
    )
    ocean_proximity_label = st.selectbox(
        '地理位置类型（Ocean Proximity）', list(OCEAN_OPTIONS.keys()),
        help='房屋或统计区块相对于海岸线的地理特征。'
    )
    predict_button = st.button('开始预测')
    ocean_proximity = OCEAN_OPTIONS[ocean_proximity_label]

model = None
preprocessor = None
y_scaler = None
model_error = None
preprocessor_error = None

if missing_dependencies:
    dep_text = '、'.join(missing_dependencies)
    error_text = (
        f'当前 Streamlit 运行的 Python 解释器是：{sys.executable}。\n'
        f'该环境缺少依赖：{dep_text}。\n'
        '请在运行 Streamlit 的同一 Python 环境中安装 scikit-learn：'
        '`pip install scikit-learn` 或 `conda install scikit-learn`。'
    )
    model_error = error_text
    preprocessor_error = error_text
else:
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f'模型文件不存在: {MODEL_PATH}')
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        model_error = f'无法加载模型: {MODEL_PATH}。错误: {e}'

    try:
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f'预处理器文件不存在: {PREPROCESSOR_PATH}')
        preprocessor = joblib.load(PREPROCESSOR_PATH)
    except Exception as e:
        preprocessor_error = f'无法加载预处理器: {PREPROCESSOR_PATH}。错误: {e}'

    try:
        if not Y_SCALER_PATH.exists():
            raise FileNotFoundError(f'y_scaler 文件不存在: {Y_SCALER_PATH}')
        y_scaler = joblib.load(Y_SCALER_PATH)
    except Exception as e:
        y_scaler_error = f'无法加载 y_scaler: {Y_SCALER_PATH}。错误: {e}'

if model_error:
    st.error(model_error)
if preprocessor_error:
    st.error(preprocessor_error)

if not model_error and not preprocessor_error and y_scaler is not None:
    st.subheader('输入信息')
    st.write({
        '经度': longitude,
        '纬度': latitude,
        '房屋中位年龄（年）': housing_median_age,
        '总房间数': total_rooms,
        '总卧室数': total_bedrooms,
        '人口数': population,
        '户数': households,
        '人均收入（万美元）': median_income,
        '距离海岸类型': ocean_proximity_label,
    })

    if predict_button:
        sample = pd.DataFrame([
            {
                'longitude': longitude,
                'latitude': latitude,
                'housing_median_age': housing_median_age,
                'total_rooms': total_rooms,
                'total_bedrooms': total_bedrooms,
                'population': population,
                'households': households,
                'median_income': median_income,
                'ocean_proximity': ocean_proximity,
            }
        ])

        try:
            X_proc = preprocessor.transform(sample)
            pred_scaled = model.predict(X_proc).reshape(-1, 1)
            prediction = y_scaler.inverse_transform(pred_scaled)[0, 0]
            st.markdown('### 预测结果')
            st.success(f'预测房价：{prediction:,.2f} 美元')
            st.write('预测结果基于已保存的随机森林模型和标准化预处理流程，仅供参考。')
        except Exception as e:
            st.error(f'预测失败: {e}')

    st.markdown('---')
    st.subheader('特征重要性')
    if FEATURE_IMPORTANCE_PATH.exists():
        st.image(str(FEATURE_IMPORTANCE_PATH), caption='随机森林特征重要性', width=700)
        st.info('随机森林模型显示“中位数收入（median_income）”是房价的最重要影响因素，说明收入水平与房价正相关。')
    else:
        st.warning('未找到特征重要性图像文件，请先运行模型比较脚本生成 feature_importances.png。')

st.sidebar.markdown('---')
st.sidebar.write('若此页面加载失败，请确保使用与 Streamlit 相同的 Python 环境运行本应用。')
st.sidebar.write('若报错“缺少 scikit-learn”，请执行：')
st.sidebar.code('pip install scikit-learn')
st.sidebar.write('或者如果使用 Conda：')
st.sidebar.code('conda install scikit-learn')
st.sidebar.write('运行方式: python -m streamlit run src/app.py')
