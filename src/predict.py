#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prediction script for single-row interactive input or CSV batch input.
Usage examples:
  python src/predict.py --model models/random_forest/rf_model.joblib --preprocessor models/random_forest/preprocessor.joblib
  python src/predict.py --model models/random_forest/rf_model.joblib --preprocessor models/random_forest/preprocessor.joblib --input data/sample.csv --output data/predictions.csv
"""
from pathlib import Path
import argparse
import pandas as pd
import joblib


def prompt_for_features(numeric_cols, categorical_cols):
    print("Enter feature values for a single house sample:")
    values = {}
    for name in numeric_cols:
        while True:
            text = input(f"  {name} (numeric): ").strip()
            if text == '':
                print("Value required. Please enter a valid number.")
                continue
            try:
                values[name] = float(text)
                break
            except ValueError:
                print("Invalid number. Please enter a numeric value.")
    for name in categorical_cols:
        text = input(f"  {name} (categorical): ").strip()
        values[name] = text if text != '' else None
    return pd.DataFrame([values])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Path to saved model joblib')
    parser.add_argument('--preprocessor', required=True, help='Path to saved preprocessor joblib')
    parser.add_argument('--input', default=None, help='Optional CSV file with rows to predict')
    parser.add_argument('--output', default=None, help='Optional CSV path to save predictions')
    args = parser.parse_args()

    model_path = Path(args.model)
    pre_path = Path(args.preprocessor)
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found: {model_path}')
    if not pre_path.exists():
        raise FileNotFoundError(f'Preprocessor not found: {pre_path}')

    model = joblib.load(model_path)
    pre = joblib.load(pre_path)

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f'Input file not found: {input_path}')
        df = pd.read_csv(input_path)
    else:
        if not hasattr(pre, 'numeric_cols') or not hasattr(pre, 'categorical_cols'):
            raise RuntimeError('Preprocessor does not expose numeric_cols/categorical_cols for interactive input')
        df = prompt_for_features(pre.numeric_cols, pre.categorical_cols)

    X_proc = pre.transform(df)
    preds = model.predict(X_proc)

    out_df = df.copy()
    out_df['prediction'] = preds

    if args.output:
        out_df.to_csv(args.output, index=False)
        print(f'Predictions saved to {args.output}')
    else:
        print('\nPrediction result:')
        print(out_df.to_string(index=False))


if __name__ == '__main__':
    main()
