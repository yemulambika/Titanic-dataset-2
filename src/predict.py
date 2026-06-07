import argparse
import pandas as pd
from pathlib import Path

from src.model import load_model, predict_single


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Titanic survival from a saved model.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/titanic_model.joblib"),
        help="Path to the saved model artifact.",
    )
    parser.add_argument("--age", type=float, required=True, help="Passenger age.")
    parser.add_argument("--fare", type=float, required=True, help="Ticket fare.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(str(args.model_path))

    X_new = pd.DataFrame({"Age": [args.age], "Fare": [args.fare]})
    prediction = predict_single(model, X_new)[0]
    label = "Survived" if int(prediction) == 1 else "Did not survive"

    print("Input:")
    print(X_new.to_string(index=False))
    print(f"Prediction: {label}")


if __name__ == "__main__":
    main()
