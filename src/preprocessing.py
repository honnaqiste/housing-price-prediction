#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""统一预处理流水线：自动识别数值与类别列，提供 fit/transform/save/load 与 特征名导出"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class Preprocessor:
    def __init__(self, numeric_impute='median', categorical_impute='most_frequent'):
        self.numeric_impute = numeric_impute
        self.categorical_impute = categorical_impute
        self.numeric_cols = None
        self.categorical_cols = None
        self.transformer = None

    def build(self, X: pd.DataFrame):
        # 自动识别列
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.numeric_impute)),
            ('scaler', StandardScaler())
        ])

        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=self.categorical_impute)),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        self.transformer = ColumnTransformer(transformers=[
            ('num', numeric_pipeline, self.numeric_cols),
            ('cat', categorical_pipeline, self.categorical_cols)
        ], remainder='drop')

    def fit(self, X: pd.DataFrame):
        if self.transformer is None:
            self.build(X)
        self.transformer.fit(X)
        return self

    def transform(self, X: pd.DataFrame):
        if self.transformer is None:
            raise RuntimeError('Preprocessor not built/fitted')
        return self.transformer.transform(X)

    def fit_transform(self, X: pd.DataFrame):
        if self.transformer is None:
            self.build(X)
        return self.transformer.fit_transform(X)

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)

    def get_feature_names_out(self):
        """Return output feature names after transformation (approximate)."""
        if self.transformer is None:
            raise RuntimeError('Preprocessor not built/fitted')

        feature_names = []
        # numeric names
        if self.numeric_cols:
            feature_names.extend(self.numeric_cols)
        # onehot names
        if self.categorical_cols:
            cat_transformer = self.transformer.named_transformers_.get('cat')
            if cat_transformer is not None:
                onehot = cat_transformer.named_steps.get('onehot')
                if onehot is not None:
                    # sklearn>=1.0
                    try:
                        o_names = onehot.get_feature_names_out(self.categorical_cols)
                        feature_names.extend(list(o_names))
                    except Exception:
                        # fallback
                        feature_names.extend(self.categorical_cols)
        return feature_names
