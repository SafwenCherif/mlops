"""Load and use the model currently designated as the champion."""

from typing import Any, Protocol, cast

import pandas as pd
from mlflow.sklearn import load_model

from mlflow_lab import configure_mlflow


class FittedChurnModel(Protocol):
    """The fitted model attributes and method used by this example."""

    feature_names_in_: Any

    def predict(self, features: pd.DataFrame) -> Any: ...


def main() -> None:
    # Connect to the same local tracking and Registry database as Lab 3/4.
    configure_mlflow()

    # The alias resolves to whichever registered version is currently champion.
    model = cast(FittedChurnModel, load_model("models:/churn-model@champion"))
    print("Model loaded:", type(model).__name__)
    print("Expected variables:", list(model.feature_names_in_))

    # Example customer with a Standard subscription.
    customer = pd.DataFrame(
        [
            {
                "age": 35,
                "salaire": 2500,
                "anciennete": 1,
                "consommation": 10,
                "abonnement_Premium": False,
                "abonnement_Standard": True,
            }
        ]
    )

    # Align feature names and order to the model, whether salary is used or not.
    customer_features: pd.DataFrame = customer.reindex(
        columns=list(model.feature_names_in_),
        fill_value=0,
    )
    prediction = model.predict(customer_features)[0]
    print("Will the customer leave? ->", "YES" if prediction == 1 else "NO")


if __name__ == "__main__":
    main()
