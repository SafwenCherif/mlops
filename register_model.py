"""Register the best Lab 3 run and assign it the champion alias."""

import mlflow
from mlflow import MlflowClient

from mlflow_lab import configure_mlflow, get_ranked_runs

MODEL_NAME = "churn-model"
CHAMPION_ALIAS = "champion"


def main() -> None:
    # Use the same local SQLite tracking/registry database as Lab 3.
    configure_mlflow()

    # Pick the best completed run with an F1 score. Tied scores favor the
    # simpler model, as in the Lab 3 results reader.
    ranked_runs = get_ranked_runs()
    if ranked_runs.empty:
        raise SystemExit("No scored runs found. Run `python train_mlflow.py` first.")

    best_run = ranked_runs.iloc[0]
    run_id = str(best_run["run_id"])
    print(
        "Best run:",
        best_run["tags.mlflow.runName"],
        "| id:",
        f"{run_id[:8]}...",
    )

    # Registering the run's saved model creates a new numbered catalog version.
    registered_version = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=MODEL_NAME,
    )
    print("Model registered:", MODEL_NAME, "| version", registered_version.version)

    # The alias is the stable name that the loader and API will use later.
    client = MlflowClient()
    client.set_registered_model_alias(
        MODEL_NAME,
        CHAMPION_ALIAS,
        registered_version.version,
    )
    print(
        f"Alias '{CHAMPION_ALIAS}' -> version",
        registered_version.version,
    )


if __name__ == "__main__":
    main()
