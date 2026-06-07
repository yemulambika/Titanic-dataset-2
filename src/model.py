from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             precision_score, recall_score)
from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def build_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(random_state=random_state, max_iter=500),
            ),
        ]
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def train_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Pipeline:
    model = build_pipeline(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "dataset",
) -> dict[str, float]:
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": report["macro avg"]["f1-score"],
    }
    print(f"Evaluation results on {dataset_name}:")
    print(f"  accuracy:  {metrics['accuracy']:.4f}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall:    {metrics['recall']:.4f}")
    print(f"  f1 score:  {metrics['f1_score']:.4f}")
    return metrics


def evaluate_cross_validation(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> dict[str, float]:
    cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "precision", "recall"],
        return_train_score=False,
    )
    metrics = {
        "cv_accuracy": float(np.mean(scores["test_accuracy"])),
        "cv_precision": float(np.mean(scores["test_precision"])),
        "cv_recall": float(np.mean(scores["test_recall"])),
    }
    print("Cross-validation results:")
    print(f"  accuracy:  {metrics['cv_accuracy']:.4f}")
    print(f"  precision: {metrics['cv_precision']:.4f}")
    print(f"  recall:    {metrics['cv_recall']:.4f}")
    return metrics


def save_model(model: Pipeline, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path_obj)
    print(f"Saved model to {path_obj.resolve()}")


def load_model(path: str) -> Pipeline:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {path_obj}")
    return load(path_obj)


def predict_single(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return model.predict(X)
