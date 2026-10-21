
from datetime import datetime, timedelta
import subprocess
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator


def run_script(script_path, error_message):
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(error_message)


def collect_live_weather():
    run_script(
        "/opt/airflow/src/ingestion/collect_live_weather.py",
        "Live weather collection failed",
    )


def calculate_rolling_average():
    run_script(
        "/opt/airflow/src/prediction/live_weather_rolling_average.py",
        "30-minute rolling average calculation failed",
    )


def run_live_regression_prediction():
    run_script(
        "/opt/airflow/src/prediction/live_regression_prediction.py",
        "Live regression prediction failed",
    )


def evaluate_live_regression():
    run_script(
        "/opt/airflow/src/prediction/evaluate_live_regression.py",
        "Live regression evaluation failed",
    )


def run_live_weather_classifier():
    run_script(
        "/opt/airflow/src/prediction/live_weather_classifier.py",
        "Live general weather classification failed",
    )


default_args = {
    "owner": "weather-ml-project",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    dag_id="weather_live_prediction",
    default_args=default_args,
    description=(
        "Live weather collection, rolling average, regression, "
        "evaluation and classification"
    ),
    start_date=datetime(2026, 9, 23),
    schedule="*/5 * * * *",
    catchup=False,
    tags=["weather", "ml", "regression", "classification"],
) as dag:

    collect_weather = PythonOperator(
        task_id="collect_live_weather",
        python_callable=collect_live_weather,
    )

    rolling_average = PythonOperator(
        task_id="calculate_30min_rolling_average",
        python_callable=calculate_rolling_average,
    )

    live_regression = PythonOperator(
        task_id="run_live_weather_prediction",
        python_callable=run_live_regression_prediction,
    )

    regression_evaluation = PythonOperator(
        task_id="evaluate_live_weather",
        python_callable=evaluate_live_regression,
    )

    general_weather = PythonOperator(
        task_id="run_live_general_weather_classification",
        python_callable=run_live_weather_classifier,
    )

    collect_weather >> rolling_average
    rolling_average >> live_regression
    live_regression >> regression_evaluation
    regression_evaluation >> general_weather
