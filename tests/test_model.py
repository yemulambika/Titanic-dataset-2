import pandas as pd

from src.model import build_pipeline, evaluate_model, predict_single


def test_build_pipeline_predicts():
    model = build_pipeline()
    X = pd.DataFrame({"Age": [22.0, 38.0, 26.0], "Fare": [7.25, 71.28, 7.92]})
    y = pd.Series([0, 1, 1])
    model.fit(X, y)
    predictions = predict_single(model, X)
    assert len(predictions) == len(X)
    assert set(predictions).issubset({0, 1})


def test_evaluate_model_returns_metrics():
    model = build_pipeline()
    X = pd.DataFrame({"Age": [22.0, 38.0, 26.0], "Fare": [7.25, 71.28, 7.92]})
    y = pd.Series([0, 1, 1])
    model.fit(X, y)
    metrics = evaluate_model(model, X, y, dataset_name="unit test")
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
