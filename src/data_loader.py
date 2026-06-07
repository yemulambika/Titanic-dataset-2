from pathlib import Path

import pandas as pd
import seaborn as sns

DEFAULT_FEATURES = ["Age", "Fare"]
DEFAULT_TARGET = "Survived"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for col in df.columns:
        normalized = col.strip().lower()
        if normalized == "age":
            mapping[col] = "Age"
        elif normalized == "fare":
            mapping[col] = "Fare"
        elif normalized in {"survived", "survival", "survived?"}:
            mapping[col] = "Survived"
    return df.rename(columns=mapping)


def load_titanic_data(csv_path: str = None) -> pd.DataFrame:
    """Load Titanic data from a local CSV or fallback to the Seaborn dataset."""
    if csv_path:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find dataset at '{csv_path}'")
        df = pd.read_csv(csv_path)
    else:
        local_csv = Path("data/titanic.csv")
        if local_csv.exists():
            df = pd.read_csv(local_csv)
        else:
            try:
                df = sns.load_dataset("titanic")
            except Exception as exc:
                raise RuntimeError(
                    "Unable to load the Titanic dataset from Seaborn and no local dataset was found at "
                    "data/titanic.csv. Provide --data-path with a valid Titanic CSV file."
                ) from exc
            if df is None:
                raise RuntimeError("Unable to load the Titanic dataset from Seaborn.")

    df = _normalize_columns(df)
    if not {DEFAULT_TARGET, *DEFAULT_FEATURES}.issubset(df.columns):
        missing = {DEFAULT_TARGET, *DEFAULT_FEATURES} - set(df.columns)
        raise ValueError(
            f"The Titanic dataset is missing required columns: {sorted(missing)}. "
            "Expected columns are Age, Fare, Survived."
        )

    return df[[*DEFAULT_FEATURES, DEFAULT_TARGET]].copy()


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df[[*DEFAULT_FEATURES, DEFAULT_TARGET]].copy()
    X = df[DEFAULT_FEATURES]
    y = df[DEFAULT_TARGET].astype(int)
    return X, y
