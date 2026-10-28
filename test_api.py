"""Exercise the Churn API over HTTP, including Pydantic validation."""

import requests

URL = "http://127.0.0.1:8000"
TIMEOUT_SECONDS = 10


# 1. Check that the service is alive.
health_response = requests.get(f"{URL}/health", timeout=TIMEOUT_SECONDS)
print("GET /health ->", health_response.json())

# 2. Customer with low usage and low seniority: the model should predict churn.
customer = {
    "age": 35,
    "salaire": 2500,
    "abonnement": "Standard",
    "anciennete": 1,
    "consommation": 10,
}
response = requests.post(
    f"{URL}/predict",
    json=customer,
    timeout=TIMEOUT_SECONDS,
)
print("POST /predict (customer 1) ->", response.status_code, response.json())

# 3. Loyal customer: the model should predict that they stay.
customer_2 = {
    "age": 50,
    "salaire": 4000,
    "abonnement": "Premium",
    "anciennete": 10,
    "consommation": 80,
}
response = requests.post(
    f"{URL}/predict",
    json=customer_2,
    timeout=TIMEOUT_SECONDS,
)
print("POST /predict (customer 2) ->", response.status_code, response.json())

# 4. Deliberately send invalid data; Pydantic should reject it with HTTP 422.
invalid_customer = dict(customer, age="abc")
response = requests.post(
    f"{URL}/predict",
    json=invalid_customer,
    timeout=TIMEOUT_SECONDS,
)
print("POST /predict (age='abc') ->", response.status_code)
print("Message:", response.json()["detail"][0]["msg"])
