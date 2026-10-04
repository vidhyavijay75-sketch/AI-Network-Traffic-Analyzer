import math
import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "model", "traffic_model.pkl")

VALID_RANGES = {
    "bandwidth_usage": (0, 1000),
    "packets_per_second": (0, 1_000_000),
    "bytes_per_second": (0, 1_000_000_000),
    "latency": (0, 10_000),
    "packet_loss": (0, 100),
    "network_utilization": (0, 100),
    "connection_count": (0, 100_000),
}

_loaded = None


class InvalidInputError(ValueError):
    pass


def load_model():
    global _loaded

    if _loaded is None:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(
                "Model file not found. Run 'python train_model.py' first."
            )

        _loaded = joblib.load(MODEL_FILE)

    return _loaded


def validate_input(data, features):
    if not isinstance(data, dict):
        raise InvalidInputError("Input must be a dictionary.")

    problems = []
    clean = {}

    for name in features:
        value = data.get(name)

        if value is None or value == "":
            problems.append(f"{name} is missing.")
            continue

        if isinstance(value, bool):
            problems.append(f"{name} must be a number.")
            continue

        try:
            number = float(value)
        except (TypeError, ValueError):
            problems.append(f"{name} must be a number.")
            continue

        if math.isnan(number) or math.isinf(number):
            problems.append(f"{name} must be a valid number.")
            continue

        low, high = VALID_RANGES[name]

        if not (low <= number <= high):
            problems.append(f"{name} must be between {low} and {high}.")
            continue

        clean[name] = number

    if problems:
        raise InvalidInputError(" ".join(problems))

    return clean


def predict_congestion(data):
    bundle = load_model()

    model = bundle["model"]
    features = bundle["features"]
    class_names = bundle["class_names"]

    clean = validate_input(data, features)

    row = pd.DataFrame(
        [[clean[name] for name in features]],
        columns=features
    )

    probabilities = model.predict_proba(row)[0]
    best = int(probabilities.argmax())

    prob_by_name = {
        class_names[int(c)]: round(float(p) * 100, 2)
        for c, p in zip(model.classes_, probabilities)
    }

    risk = prob_by_name["High"] + prob_by_name["Congestion"]

    return {
        "congestion_level": class_names[int(model.classes_[best])],
        "confidence": round(float(probabilities[best]) * 100, 2),
        "risk_percentage": round(risk, 2),
        "probabilities": prob_by_name,
    }


if __name__ == "__main__":

    samples = {
        "Quiet network": {
            "bandwidth_usage": 12,
            "packets_per_second": 2500,
            "bytes_per_second": 1500000,
            "latency": 10,
            "packet_loss": 0.1,
            "network_utilization": 12,
            "connection_count": 60,
        },

        "Busy network": {
            "bandwidth_usage": 52,
            "packets_per_second": 9500,
            "bytes_per_second": 6500000,
            "latency": 38,
            "packet_loss": 0.6,
            "network_utilization": 52,
            "connection_count": 310,
        },

        "Overloaded network": {
            "bandwidth_usage": 95,
            "packets_per_second": 17000,
            "bytes_per_second": 11800000,
            "latency": 180,
            "packet_loss": 6.5,
            "network_utilization": 95,
            "connection_count": 850,
        },
    }

    for title, values in samples.items():
        result = predict_congestion(values)

        print(
            f"{title:20s} -> "
            f"{result['congestion_level']:10s} "
            f"confidence {result['confidence']:6.2f}% "
            f"risk {result['risk_percentage']:6.2f}%"
        )

    print("\nInvalid input test:")

    try:
        predict_congestion({
            "bandwidth_usage": -5,
            "latency": "abc"
        })
    except InvalidInputError as error:
        print("Rejected correctly ->", error)