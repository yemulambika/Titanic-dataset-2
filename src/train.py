import argparse
from pathlib import Path

from src.data_loader import load_titanic_data, prepare_features
from src.model import (
    split_data,
    train_pipeline,
    evaluate_model,
    evaluate_cross_validation,
    save_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Titanic survival prediction model."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Optional local CSV path for the Titanic dataset.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/titanic_model.joblib"),
        help="Output path for the trained model artifact.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data reserved for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_titanic_data(csv_path=str(args.data_path) if args.data_path else None)
    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    model = train_pipeline(X_train, y_train, random_state=args.random_state)
    evaluate_model(model, X_test, y_test, dataset_name="test set")
    evaluate_cross_validation(model, X, y, cv=args.cv_folds)
    save_model(model, str(args.model_path))


if __name__ == "__main__":
    main()
