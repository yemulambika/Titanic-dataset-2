import pytest

from src.data_loader import load_titanic_data, prepare_features


def test_load_titanic_data_default():
    df = load_titanic_data()
    assert "Age" in df.columns
    assert "Fare" in df.columns
    assert "Survived" in df.columns
    X, y = prepare_features(df)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 2


def test_load_titanic_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_titanic_data(csv_path="nonexistent-file.csv")
