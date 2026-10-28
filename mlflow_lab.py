"""Shared configuration and helpers for the Lab 3 MLflow scripts."""

from pathlib import Path

import mlflow
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "churn_data.csv"
TRACKING_DB_PATH = PROJECT_DIR / "mlflow.db"
EXPERIMENT_NAME = "churn-prediction"
ALL_FEATURES = ["age", "salaire", "abonnement", "anciennete", "consommation"]


def configure_mlflow() -> None:
    """Keep runs and artifacts beside this script, regardless of the shell cwd."""
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB_PATH.as_posix()}")

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        artifact_dir = PROJECT_DIR / "mlruns" / EXPERIMENT_NAME
        mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=artifact_dir.as_uri(),
        )

    mlflow.set_experiment(EXPERIMENT_NAME)


def get_ranked_runs() -> pd.DataFrame:
    """Return runs with a recorded F1 score, best score first.

    Ties are resolved in favor of fewer original input columns, then higher
    recall and precision. This makes the choice reproducible and favors the
    simpler model when validation results are identical.
    """
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=["metrics.f1 DESC"],
    )
    if runs.empty or "metrics.f1" not in runs.columns:
        return runs

    ranked = runs.loc[runs["metrics.f1"].notna()].copy()
    if ranked.empty:
        return ranked

    feature_counts = ranked.get("params.features", pd.Series(index=ranked.index, dtype="object"))
    ranked["_feature_count"] = feature_counts.fillna("").map(
        lambda value: len(value.split(",")) if value else float("inf")
    )
    for metric in ("metrics.recall", "metrics.precision"):
        if metric not in ranked.columns:
            ranked[metric] = 0.0
        ranked[metric] = ranked[metric].fillna(0.0)

    return ranked.sort_values(
        by=["metrics.f1", "_feature_count", "metrics.recall", "metrics.precision"],
        ascending=[False, True, False, False],
        kind="stable",
    ).drop(columns="_feature_count")
