# Lab 3 — Reflection

### Which model is best, and how can I prove it?

I compare the runs using F1 rather than accuracy alone, because churn is a rare class and accuracy can hide missed departures. MLflow stores each trial's parameters and test metrics, so I can inspect and sort the results in the UI or with `read_runs.py`. If two trials tie, this project prefers the model using fewer input columns.

### Can I retrieve the exact model from Trial 1?

Yes. Each run stores its own model artifact. With that run's ID, `mlflow.sklearn.load_model("runs:/<run_id>/model")` reloads the trained model without retraining it. `reload_model.py` demonstrates this using the highest-ranked run.

### Can a classmate reproduce the work on another machine?

They can clone the code, install the packages in `requirements.txt`, and rerun the training scripts. MLflow records the run parameters, metrics, and model artifacts locally, but `mlflow.db` and `mlruns/` are intentionally excluded from Git. A classmate therefore creates their own run history by rerunning the scripts; exact environment reproducibility would be improved further with a pinned environment or Docker image.
