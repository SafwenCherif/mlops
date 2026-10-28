"""Print the recorded runs and identify the highest-ranked trial."""

import mlflow

from mlflow_lab import configure_mlflow, get_ranked_runs


def main() -> None:
    configure_mlflow()
    runs = mlflow.search_runs(
        experiment_names=["churn-prediction"],
        order_by=["metrics.f1 DESC"],
    )
    if runs.empty:
        raise SystemExit("No runs found. Train runs first with `python train_mlflow.py`.")

    columns = [
        "tags.mlflow.runName",
        "params.n_estimators",
        "params.max_depth",
        "params.features",
        "metrics.accuracy",
        "metrics.precision",
        "metrics.recall",
        "metrics.f1",
    ]
    available_columns = [column for column in columns if column in runs.columns]
    print(runs[available_columns].to_string(index=False))

    ranked = get_ranked_runs()
    if ranked.empty:
        print("\nNo run has an F1 metric yet; run `python train_mlflow.py`.")
        return

    best = ranked.iloc[0]
    print(
        f"\nBest run: {best['tags.mlflow.runName']} "
        f"| id: {best['run_id'][:8]}... "
        f"| f1: {best['metrics.f1']:.3f}"
    )


if __name__ == "__main__":
    main()
