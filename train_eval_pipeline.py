#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train & Evaluate: Intra-city and Cross-city experiments

- Loads harmonized CSVs from data_out/
- Builds models: XGBoost, RandomForest, Logistic Regression baseline
- Evaluates with ROC-AUC, PR-AUC, F1, Accuracy
- Saves ROC curves and XGBoost feature importance plots
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from plotting_utils import plot_roc_curves, plot_feature_importance

RANDOM_STATE = 42

FEATURES = [
    "speed","reference_speed","travel_time_minutes",
    "weekday","time_sin","time_cos"
]
TARGET = "incident"

def _dropna_cols(df, cols):
    return df.dropna(subset=[c for c in cols if c in df.columns])

def load_city(path):
    df = pd.read_csv(path)
    df = _dropna_cols(df, FEATURES+[TARGET])
    # Some datasets might have missing weekday: fill with -1
    if "weekday" in df.columns:
        df["weekday"] = df["weekday"].fillna(-1)
    return df

def train_eval_single(df, model_name="xgboost"):
    X = df[FEATURES].values
    y = df[TARGET].values.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

    if model_name == "xgboost":
        clf = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss"
        )
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:,1]
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:,1]
    else:  # logistic regression baseline
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=200, n_jobs=-1, random_state=RANDOM_STATE)
        clf.fit(X_train_s, y_train)
        y_score = clf.predict_proba(X_test_s)[:,1]

    y_pred = (y_score >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "pr_auc": float(average_precision_score(y_test, y_score)),
        "f1": float(f1_score(y_test, y_pred)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }
    fpr, tpr, _ = roc_curve(y_test, y_score)
    return clf, metrics, (fpr, tpr)

def cross_city_train_eval(train_df, test_df, model_name="xgboost"):
    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values.astype(int)
    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values.astype(int)

    if model_name == "xgboost":
        clf = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss"
        )
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:,1]
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        y_score = clf.predict_proba(X_test)[:,1]
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        clf = LogisticRegression(max_iter=200, n_jobs=-1, random_state=RANDOM_STATE)
        clf.fit(X_train_s, y_train)
        y_score = clf.predict_proba(X_test_s)[:,1]

    y_pred = (y_score >= 0.5).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "pr_auc": float(average_precision_score(y_test, y_score)),
        "f1": float(f1_score(y_test, y_pred)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
    }
    fpr, tpr, _ = roc_curve(y_test, y_score)
    return clf, metrics, (fpr, tpr)

def main():
    os.makedirs("results", exist_ok=True)

    # Load harmonized datasets
    df_fl = load_city("data_out/florida_harmonized.csv")
    df_ta = load_city("data_out/tallinn_harmonized.csv")

    # Intra-city evaluations
    curves = []
    metrics_all = {}

    for name, df in [("Florida", df_fl), ("Tallinn", df_ta)]:
        clf, metrics, (fpr, tpr) = train_eval_single(df, model_name="xgboost")
        metrics_all[f"intra_{name.lower()}"] = metrics
        curves.append({"fpr": fpr, "tpr": tpr, "auc": metrics["roc_auc"], "label": f"Intra {name} (XGB)"})

    # Cross-city (train on Florida, test on Tallinn) and vice versa
    clf_ft, m_ft, (fpr_ft, tpr_ft) = cross_city_train_eval(df_fl, df_ta, model_name="xgboost")
    metrics_all["cross_florida_to_tallinn"] = m_ft
    curves.append({"fpr": fpr_ft, "tpr": tpr_ft, "auc": m_ft["roc_auc"], "label": "Florida→Tallinn (XGB)"})

    clf_tf, m_tf, (fpr_tf, tpr_tf) = cross_city_train_eval(df_ta, df_fl, model_name="xgboost")
    metrics_all["cross_tallinn_to_florida"] = m_tf
    curves.append({"fpr": fpr_tf, "tpr": tpr_tf, "auc": m_tf["roc_auc"], "label": "Tallinn→Florida (XGB)"})

    # Save metrics
    with open("results/metrics.json", "w") as f:
        json.dump(metrics_all, f, indent=2)
    print("[OK] results/metrics.json")
    for k,v in metrics_all.items():
        print(k, v)

    # ROC curves figure
    plot_roc_curves(curves, out_pdf="results/roc_curves.pdf")
    print("[OK] results/roc_curves.pdf (and .png)")

    # Feature importance from the Florida intra-city XGB as example
    if hasattr(clf, "feature_importances_"):
        plot_feature_importance(clf.feature_importances_, FEATURES, out_pdf="results/feature_importance.pdf", top_k=len(FEATURES))
        print("[OK] results/feature_importance.pdf (and .png)")

if __name__ == "__main__":
    main()
