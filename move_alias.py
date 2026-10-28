"""Demonstrate promoting Trial 2, then rolling the champion alias back."""

import mlflow
from mlflow import MlflowClient

from mlflow_lab import configure_mlflow

MODEL_NAME = "churn-model"
CHAMPION_ALIAS = "champion"


def main() -> None:
    configure_mlflow()
    client = MlflowClient()

    # Find a scored Trial 2 run, which is the deliberately weaker model.
    trial_2_runs = mlflow.search_runs(
        experiment_names=["churn-prediction"],
        filter_string="tags.mlflow.runName = 'trial-2'",
        order_by=["metrics.f1 DESC"],
    )
    if trial_2_runs.empty or trial_2_runs["metrics.f1"].dropna().empty:
        raise SystemExit("No scored trial-2 run found. Run `python train_mlflow.py` first.")

    trial_2_run_id = str(trial_2_runs.loc[trial_2_runs["metrics.f1"].notna()].iloc[0]["run_id"])

    # Save the current champion so we can restore it after the demonstration.
    previous_champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    previous_version = str(previous_champion.version)
    print("Current champion: version", previous_version)

    # Registering this different run creates another numbered version.
    trial_2_version = mlflow.register_model(
        model_uri=f"runs:/{trial_2_run_id}/model",
        name=MODEL_NAME,
    )
    new_version = str(trial_2_version.version)
    print("New Trial 2 version created:", new_version)

    # Move the alias to Trial 2, then always restore the original version.
    try:
        client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, new_version)
        moved_champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
        print("After the move: champion -> version", moved_champion.version)
    finally:
        client.set_registered_model_alias(
            MODEL_NAME,
            CHAMPION_ALIAS,
            previous_version,
        )

    restored_champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    print("After rollback: champion -> version", restored_champion.version)


if __name__ == "__main__":
    main()
