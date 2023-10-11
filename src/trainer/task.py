from __future__ import annotations

import argparse
import glob
import json
import math
import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "tip_bin"
TARGET_LABELS = ["tip<20%", "tip>=20%"]


def sanitize(path: str) -> str:
    return path.replace("gs://", "/gcs/", 1) if path and path.startswith("gs://") else path


def get_dataframe(path: str) -> pd.DataFrame:
    if os.path.isdir(path):  # base data directory is passed
        files = glob.glob(f"{path}/*.csv")
    elif "*" in path:  # a glob expression is passed
        files = glob.glob(path)
    else:  # single file is passed
        files = [path]
    dfs = (pd.read_csv(f, header=0) for f in files)
    return pd.concat(dfs, ignore_index=True)


def create_datasets(training_data_dir: str, validation_data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Creates training and validation datasets."""

    train_dataset = get_dataframe(training_data_dir)

    if validation_data_dir:
        return train_dataset, get_dataframe(validation_data_dir)
    else:
        return train_test_split(train_dataset, test_size=.25, random_state=42)


# Newer versions of scikit-learn generate `Infinity` (instead of `max(y_score) + 1`) as the first threshold when the
# `roc_curve` is calculated. Since we're storing the thresholds in JSON format, and JSON can't handle `Infinity` we're
# reverting back to the old behaviour by replacing `Infinity with `max(y_score) + 1`.
def strip_infinity(thresholds: list[float]):
    """JSON can't handle infinity, replacing that to work around that limitation"""
    if math.inf in thresholds:
        replacement = max([t for t in thresholds if t != math.inf]) + 1
        return [replacement if t == math.inf else t for t in thresholds]
    else:
        return thresholds


def log_metrics(y_pred: pd.Series, y_true: pd.Series, output_dir: str):
    curve = roc_curve(y_score=y_pred, y_true=y_true)
    auc = roc_auc_score(y_score=y_pred, y_true=y_true)
    cm = confusion_matrix(labels=[False, True], y_pred=y_pred, y_true=y_true)

    with open(f"{output_dir}/metrics.json", "w") as f:    
        metrics = {"auc": auc}
        metrics["confusion_matrix"] = {}
        metrics["confusion_matrix"]["categories"] = TARGET_LABELS
        metrics["confusion_matrix"]["matrix"] = cm.tolist()
        metrics["roc_curve"] = {}
        metrics["roc_curve"]["fpr"] = curve[0].tolist()
        metrics["roc_curve"]["tpr"] = curve[1].tolist()
        metrics["roc_curve"]["thresholds"] = strip_infinity(curve[2].tolist())
        json.dump(metrics, f, indent=2)


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(TARGET_COLUMN, axis=1), df[TARGET_COLUMN]


def train(training_data_dir: str, validation_data_dir: str, output_dir: str) -> float:
    train_df, val_df = create_datasets(training_data_dir, validation_data_dir)

    X_train, y_train = split(train_df)
    X_test, y_test = split(val_df)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, f"{output_dir}/model.joblib")

    y_pred = model.predict(X_test)
    log_metrics(y_pred, y_test, output_dir)

    return model.score(X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data-dir", type=str, default=os.getenv("AIP_TRAINING_DATA_URI", "data"))
    parser.add_argument("--validation-data-dir", type=str, default=os.getenv("AIP_VALIDATION_DATA_URI", None))
    parser.add_argument("--output-dir", type=str, default=os.getenv("AIP_MODEL_DIR", "outputs"))

    args = parser.parse_args()

    train(
        sanitize(args.training_data_dir),
        sanitize(args.validation_data_dir),
        sanitize(args.output_dir))
