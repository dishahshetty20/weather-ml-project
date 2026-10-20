FROM apache/airflow:2.9.2

USER airflow

RUN python -m pip install --no-cache-dir --default-timeout=300 --retries 10 \
    numpy==1.26.4 \
    pandas==2.1.4 \
    scipy==1.17.1 \
    joblib==1.6.0 \
    cloudpickle==3.1.2 \
    scikit-learn==1.9.1 \
    threadpoolctl \
    narwhals \
    requests==2.31.0 \
    psycopg2-binary==2.9.9

RUN python -m pip install --no-cache-dir --default-timeout=300 --retries 10 --no-deps xgboost==3.4.1
