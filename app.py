"""FastAPI service that predicts churn with the registered champion model."""

from typing import Any, Literal, Protocol, cast

import pandas as pd
from fastapi import FastAPI
from mlflow.sklearn import load_model
from pydantic import BaseModel

from mlflow_lab import configure_mlflow


class FittedChurnModel(Protocol):
    """The fitted model interface required by the prediction endpoint."""

    feature_names_in_: Any

    def predict(self, features: pd.DataFrame) -> Any: ...


app = FastAPI(title="Churn API")

# Configure the local Lab 3 database and load the current champion once when
# the API process starts, rather than reloading the model for every request.
configure_mlflow()
model = cast(FittedChurnModel, load_model("models:/churn-model@champion"))


class Customer(BaseModel):
    """Input fields accepted by the churn prediction endpoint."""

    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float


@app.get("/health")
def health() -> dict[str, str]:
    """Return a simple response for health checks."""
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer) -> dict[str, int | str]:
    """Validate a customer, align its features, and return the prediction."""
    row = pd.DataFrame([customer.model_dump()])
    row = pd.get_dummies(row)
    customer_features: pd.DataFrame = row.reindex(
        columns=list(model.feature_names_in_),
        fill_value=0,
    )

    result = int(model.predict(customer_features)[0])
    message = "The customer will leave" if result == 1 else "The customer stays"
    return {"churn": result, "message": message}
