# Lab 3 — Churn experiment tracking with MLflow

This folder contains the complete Lab 3 workflow. It uses `churn_data.csv` and keeps the MLflow tracking database and model artifacts locally in this project.

## 1. Create an environment and install dependencies

Use Python 3.12 or 3.13 for the smoothest compatibility with the MLflow/scikit-learn stack.

### Linux / macOS

```bash
cd /path/to/tp1
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your system provides Python 3.13 instead, replace `python3.12` with `python3.13`.

### Windows PowerShell

```powershell
cd C:\path\to\tp1
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Check the install:

```bash
python -m mlflow --version
```

If `churn_data.csv` is missing, generate it with `python generate_data.py`.

## 2. Run the introductory single trial (Lab Step 1)

```bash
python train_mlflow.py --single-run
```

This records Trial 1 with its parameters, accuracy, and saved model. It corresponds to the first, accuracy-only run in the lab instructions.

## 3. Train the three comparable trials (Lab Step 3)

```bash
python train_mlflow.py
```

This creates three new runs (`trial-1`, `trial-2`, and `trial-3`) with accuracy, precision, recall, F1, parameters, and a saved model per run. Runs are append-only: each execution creates new runs rather than replacing older ones.

The scripts configure the SQLite tracking store at `mlflow.db` and artifacts under `mlruns/` beside the scripts. You can launch them from another working directory; the data and tracking paths still resolve to this project.

## 4. Browse and compare runs

From this project directory, start the local UI:

```bash
python -m mlflow ui --host 127.0.0.1 --port 5000
```

Open <http://127.0.0.1:5000>, select **Model training**, then the `churn-prediction` experiment and its runs. Add the accuracy, precision, recall, F1, and parameter columns to the table to compare runs. Stop the UI with `Ctrl+C`.

If port 5000 is already occupied, use `--port 5001` and open <http://127.0.0.1:5001>.

## 5. Read results and reload the best model

```bash
python read_runs.py
python reload_model.py
```

The scripts ignore runs without an F1 score when selecting the best run. Ranking is by F1 first; ties prefer fewer input columns, then higher recall and precision. `reload_model.py` fetches that exact run's model artifact and predicts for an example customer.

## 6. Optional automatic logging

```bash
python train_autolog.py
```

This creates a `trial-auto` run and demonstrates scikit-learn autologging. Its training metrics are not used as the held-out test F1 in the manual trial comparison.

## Files

- `mlflow_lab.py` — shared paths, experiment setup, and deterministic best-run ranking.
- `train_mlflow.py` — single introductory run (`--single-run`) or the three manual trials.
- `read_runs.py` — prints recorded runs and the selected best run.
- `reload_model.py` — reloads the selected model from its run and predicts.
- `train_autolog.py` — optional scikit-learn autolog demonstration.
- `requirements.txt` — project dependencies.

`mlflow.db`, `mlruns/`, and the virtual environment are generated locally and excluded from Git. To start with a clean tracking history, stop any MLflow UI process and remove those generated paths before running the scripts again.
