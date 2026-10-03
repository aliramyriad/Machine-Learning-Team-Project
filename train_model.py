"""Stage 1: train and evaluate the "will this song be liked?" classifier.

Trains on Jan-Jul plays and tests on Aug-Sep plays (a time split, so we test on
the "future" like a real recommender would), compares against simple
baselines, then refits on all data and saves the model.

Usage:
    python train_model.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from features import CATEGORICAL, CONTEXTS, FEATURES, NUMERIC, build_features, load_plays

TEST_START = pd.Timestamp("2026-08-01", tz="UTC")


def gradient_boosting():
    encode = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CATEGORICAL)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    model = HistGradientBoostingClassifier(
        categorical_features=CATEGORICAL,
        learning_rate=0.05,
        max_iter=300,
        early_stopping=True,
        random_state=0,
    )
    return make_pipeline(encode, model)


def logistic_regression():
    encode = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
    ])
    return make_pipeline(encode, LogisticRegression(max_iter=1000))


def scores(y, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, prob) if len(set(y)) > 1 else float("nan"),
    }


def main():
    df = build_features(load_plays())
    train, test = df[df["ts"] < TEST_START], df[df["ts"] >= TEST_START]
    X_train, y_train = train[FEATURES], train["liked"]
    X_test, y_test = test[FEATURES], test["liked"]
    print(f"Train: {len(train):,} plays   Test: {len(test):,} plays   "
          f"Liked rate: {df['liked'].mean():.1%}\n")

    models = {
        "baseline (always majority)": DummyClassifier(strategy="most_frequent"),
        "logistic regression": logistic_regression(),
        "gradient boosting": gradient_boosting(),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = scores(y_test, model.predict_proba(X_test)[:, 1])
    print(pd.DataFrame(results).T.round(3).to_string(), "\n")

    best = models["gradient boosting"]
    prob = best.predict_proba(X_test)[:, 1]
    by_context = {ctx: scores(y_test[test["context"] == ctx], prob[test["context"] == ctx])
                  for ctx in CONTEXTS}
    print("Gradient boosting, test set by playlist context:")
    print(pd.DataFrame(by_context).T.round(3).to_string(), "\n")

    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=0)
    importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False)
    print("Most useful features (drop in ROC-AUC when shuffled):")
    print(importance.head(10).round(4).to_string(), "\n")

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump({"test": results, "test_by_context": by_context,
                   "feature_importance": importance.round(4).to_dict()}, f, indent=2)

    # Refit on everything for building playlists.
    final = gradient_boosting().fit(df[FEATURES], df["liked"])
    Path("models").mkdir(exist_ok=True)
    joblib.dump(final, "models/like_model.joblib")
    print("Saved models/like_model.joblib and outputs/metrics.json")


if __name__ == "__main__":
    main()
