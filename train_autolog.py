"""Bonus: train a model while letting MLflow log it automatically."""

import mlflow
import pandas as pd
from mlflow.sklearn import autolog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from mlflow_lab import ALL_FEATURES, DATA_PATH, configure_mlflow


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Run `python generate_data.py` first."
        )

    configure_mlflow()
    df = pd.read_csv(DATA_PATH)
    X = pd.get_dummies(df[ALL_FEATURES], drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        df["churn"],
        test_size=0.2,
        random_state=42,
    )

    autolog(serialization_format="cloudpickle")
    with mlflow.start_run(run_name="trial-auto"):
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            random_state=42,
        )
        model.fit(X_train, y_train)
        print("test score :", model.score(X_test, y_test))


if __name__ == "__main__":
    main()
