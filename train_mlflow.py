"""Train the churn trials and record their parameters, metrics, and models."""

import argparse

import mlflow
import pandas as pd
from mlflow.sklearn import log_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from mlflow_lab import ALL_FEATURES, DATA_PATH, configure_mlflow


def run_trial(
    name: str,
    n_estimators: int,
    max_depth: int,
    columns: list[str],
    df: pd.DataFrame,
    *,
    accuracy_only: bool = False,
) -> None:
    """Fit one Random Forest and record it as a distinct MLflow run."""
    X = pd.get_dummies(df[columns], drop_first=True)
    y = df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    with mlflow.start_run(run_name=name):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mlflow.log_params(
            {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "features": ",".join(columns),
            }
        )
        accuracy = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", accuracy)

        if not accuracy_only:
            mlflow.log_metrics(
                {
                    "precision": precision_score(y_test, predictions, zero_division="warn"),
                    "recall": recall_score(y_test, predictions, zero_division="warn"),
                    "f1": f1_score(y_test, predictions, zero_division="warn"),
                }
            )

        # Explicitly use cloudpickle for compatibility with MLflow 3's
        # scikit-learn model serialization and this locally-created model.
        log_model(
            model,
            name="model",
            serialization_format="cloudpickle",
        )

        if accuracy_only:
            print(f"{name} -> accuracy : {accuracy:.3f}")
        else:
            f1 = f1_score(y_test, predictions, zero_division="warn")
            print(f"{name} -> accuracy : {accuracy:.3f} | f1 : {f1:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--single-run",
        action="store_true",
        help="Run only Trial 1 and log accuracy, matching Lab 3 Step 1.",
    )
    args = parser.parse_args()

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Run `python generate_data.py` first."
        )

    configure_mlflow()
    df = pd.read_csv(DATA_PATH)

    if args.single_run:
        run_trial("trial-1", 100, 5, ALL_FEATURES, df, accuracy_only=True)
        return

    run_trial("trial-1", 100, 5, ALL_FEATURES, df)
    run_trial("trial-2", 50, 3, ALL_FEATURES, df)
    run_trial(
        "trial-3",
        100,
        5,
        ["age", "abonnement", "anciennete", "consommation"],
        df,
    )


if __name__ == "__main__":
    main()
