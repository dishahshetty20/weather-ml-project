from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


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
        raise RuntimeError("Live prediction failed")


default_args = {
    "owner": "weather-ml-project",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="weather_live_prediction",
    default_args=default_args,
    description="Run live Bengaluru weather prediction",
    start_date=datetime(2026, 9, 23),
    schedule="*/15 * * * *",
    catchup=False,
    tags=["weather", "ml", "prediction"],
) as dag:

    live_prediction = PythonOperator(
        task_id="run_live_prediction",
        python_callable=run_live_prediction,
    )