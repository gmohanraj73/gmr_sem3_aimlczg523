"""Generate API traffic so the Prometheus/Grafana dashboards show live metrics.

Sends a series of /predict requests using a mix of sample patients.
Run (with the API up, e.g. via docker-compose.monitoring.yml):
    python scripts/load_test.py
"""
from __future__ import annotations

import argparse
import time

import requests

# Five valid raw-encoded patient records spanning the risk spectrum.
SAMPLE_PATIENTS = [
    {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
     "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6},
    {"age": 67, "sex": 1, "cp": 4, "trestbps": 120, "chol": 229, "fbs": 0,
     "restecg": 2, "thalach": 129, "exang": 1, "oldpeak": 2.6, "slope": 2, "ca": 2, "thal": 7},
    {"age": 37, "sex": 1, "cp": 3, "trestbps": 130, "chol": 250, "fbs": 0,
     "restecg": 0, "thalach": 187, "exang": 0, "oldpeak": 3.5, "slope": 3, "ca": 0, "thal": 3},
    {"age": 41, "sex": 0, "cp": 2, "trestbps": 130, "chol": 204, "fbs": 0,
     "restecg": 2, "thalach": 172, "exang": 0, "oldpeak": 1.4, "slope": 1, "ca": 0, "thal": 3},
    {"age": 56, "sex": 1, "cp": 4, "trestbps": 130, "chol": 256, "fbs": 1,
     "restecg": 2, "thalach": 142, "exang": 1, "oldpeak": 0.6, "slope": 2, "ca": 1, "thal": 6},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Load-test the /predict endpoint.")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--n", type=int, default=50, help="number of requests")
    parser.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
    args = parser.parse_args()

    endpoint = f"{args.url}/predict"
    ok, failed = 0, 0
    for i in range(args.n):
        # Deterministic round-robin (no RNG) keeps the run reproducible.
        patient = SAMPLE_PATIENTS[i % len(SAMPLE_PATIENTS)]
        try:
            resp = requests.post(endpoint, json=patient, timeout=5)
            resp.raise_for_status()
            body = resp.json()
            ok += 1
            print(f"Request {i + 1:>3}: prediction={body['prediction']} "
                  f"prob={body['probability']:.3f} risk={body['risk_level']}")
        except Exception as exc:  # noqa: BLE001 - report and continue under load
            failed += 1
            print(f"Request {i + 1:>3}: FAILED ({exc})")
        time.sleep(args.delay)

    print(f"\nDone: {ok} ok, {failed} failed out of {args.n}.")


if __name__ == "__main__":
    main()
