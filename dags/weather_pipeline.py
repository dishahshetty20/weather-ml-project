from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


# ============================================================
# LIVE PREDICTION
# ============================================================

def run_live_prediction():

    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "/opt/airflow/src/prediction/live_prediction.py"
        ],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:

        print(result.stderr)

        raise RuntimeError(
            "Live prediction failed"
        )


# ============================================================
# LIVE PREDICTION EVALUATION
# ============================================================

def evaluate_live_predictions():

    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "/opt/airflow/src/prediction/evaluate_live_predictions.py"
        ],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode != 0:

        print(result.stderr)

        raise RuntimeError(
            "Live prediction evaluation failed"
        )


# ============================================================
# AIRFLOW DEFAULT ARGUMENTS
# ============================================================

default_args = {
    "owner": "weather-ml-project",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


# ============================================================
# DAG
# ============================================================

with DAG(
    dag_id="weather_live_prediction",
    default_args=default_args,
    description="Run live Bengaluru weather prediction and evaluation",
    start_date=datetime(2026, 9, 23),
    schedule="*/15 * * * *",
    catchup=False,
    tags=[
        "weather",
        "ml",
        "prediction",
        "evaluation"
    ],
) as dag:

    live_prediction = PythonOperator(
        task_id="run_live_prediction",
        python_callable=run_live_prediction,
    )

    live_evaluation = PythonOperator(
        task_id="evaluate_live_predictions",
        python_callable=evaluate_live_predictions,
    )

    live_prediction >> live_evaluation