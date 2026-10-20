\# Weather ML Project — Bengaluru



A weather data engineering and machine learning project that collects historical weather data, trains prediction models, and automates live weather prediction and evaluation using Apache Airflow.



\## Project Overview



The project uses historical Bengaluru weather data from 2021 to 2025 to develop machine learning models and a scheduled pipeline for live weather predictions.



\*\*Key features\*\*

\- Historical weather data ingestion using the Open-Meteo API.

\- Data preprocessing and storage.

\- Regression models for predicting numerical weather variables.

\- Classification models for general weather categories.

\- Automated live predictions and evaluation using Apache Airflow.

\- Performance comparison using evaluation metrics and graphs.



\## Technology Stack



\- \*\*Language:\*\* Python

\- \*\*Machine Learning:\*\* Scikit-learn, XGBoost

\- \*\*Data Processing:\*\* Pandas, NumPy

\- \*\*Data Visualization:\*\* Matplotlib

\- \*\*Data Source:\*\* Open-Meteo API

\- \*\*Database:\*\* PostgreSQL

\- \*\*Workflow Orchestration:\*\* Apache Airflow

\- \*\*Environment:\*\* Docker, Python virtual environment



\## Historical Dataset



Historical weather data covers January 2021 through December 2025.



\- \*\*Records:\*\* 43,824 hourly observations

\- \*\*Columns:\*\* 25

\- \*\*Location:\*\* Bengaluru

\- \*\*Source:\*\* Open-Meteo historical weather API



The data is used for preprocessing, model training, and historical performance evaluation.



\## Machine Learning Models



\### Regression



Regression models predict numerical weather variables, including:



\- Temperature

\- Relative humidity

\- Wind speed

\- Wind gusts

\- Atmospheric pressure

\- Dew point

\- Precipitation

\- Cloud cover

\- Vapour pressure deficit



Models include Linear Regression and XGBoost.



\### Classification



Random Forest and XGBoost are used for general weather classification.



Model evaluation includes accuracy, precision, recall, and F1-score. Regression evaluation includes MAE, RMSE, and R².



Actual performance depends on the evaluation dataset and sample size.



\## Live Prediction Pipeline



Apache Airflow runs the `weather\_live\_prediction` DAG every 15 minutes.



The workflow executes these tasks in sequence:



1\. \*\*Live Weather Prediction:\*\* Generates regression predictions using the live prediction script.

2\. \*\*Regression Evaluation:\*\* Retrieves weather observations where available and evaluates previous predictions.

3\. \*\*Weather Classification:\*\* Runs the general weather classification model.



The pipeline uses retries to help recover from temporary task failures.



\## Project Structure



```text

weather-ml-project/

├── dags/

│   └── weather\_pipeline.py

├── data/

│   ├── raw/

│   └── live/

├── evaluation/

│   ├── regression/

│   └── live\_classification/

├── models/

├── ml/

├── scripts/

├── sql/

├── src/

│   ├── ingestion/

│   ├── preprocessing/

│   └── prediction/

├── config/

├── docker/

├── docker-compose.yml

├── Dockerfile

└── README.md

```



\## Running the Project



\### 1. Clone the repository



```bash

git clone https://github.com/dishahshetty20/weather-ml-project.git

cd weather-ml-project

```



\### 2. Create and activate a virtual environment



On Windows PowerShell:



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



Install dependencies if a project requirements file is available:



```powershell

pip install -r requirements.txt

```



\### 3. Start the Docker services



Ensure Docker Desktop is running, then execute:



```powershell

docker compose up -d

```



Follow the project's existing database and Airflow configuration before running the pipeline.



\### 4. Access Airflow



Open:



http://localhost:8080



Locate the `weather\_live\_prediction` DAG and check its task status and execution logs.



\## Evaluation and Results



The project generates evaluation CSV files and graphs for comparing model performance.



Important output files include:



\- `evaluation/regression/live\_metrics.csv`

\- `evaluation/regression/live\_best\_model\_by\_target.csv`

\- `evaluation/live\_classification/live\_classification\_metrics.csv`

\- `evaluation/live\_classification/live\_classification\_metrics.png`



Live evaluation results depend on the availability of observations and the number of evaluated samples. Small sample sizes should be interpreted cautiously.



\## Future Improvements



\- Improve live observation matching and data-source tracking.

\- Collect more verified live evaluation samples.

\- Compare models using consistent validation datasets.

\- Improve weather classification recall.

\- Add monitoring dashboards and alerts.



\## Author



Developed as a weather data engineering and machine learning project.



