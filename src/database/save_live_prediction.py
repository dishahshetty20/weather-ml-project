import psycopg2


DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "weather_db"
DB_USER = "weather_user"
DB_PASSWORD = "weather_password"


def save_prediction(
    prediction_time,
    rf_prediction,
    rf_probability,
    xgb_prediction,
    xgb_probability
):

    print("Connecting to PostgreSQL...")

    connection = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

    cursor = connection.cursor()

    print("PostgreSQL connection successful!")

    create_table_query = """
    CREATE TABLE IF NOT EXISTS live_predictions (
        id SERIAL PRIMARY KEY,
        prediction_time TIMESTAMP NOT NULL,
        rf_prediction INTEGER,
        rf_probability DOUBLE PRECISION,
        xgb_prediction INTEGER,
        xgb_probability DOUBLE PRECISION,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    cursor.execute(create_table_query)

    # Prevent duplicate predictions for the same prediction time
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS
        unique_live_prediction_time
        ON live_predictions (prediction_time);
    """)

    insert_query = """
    INSERT INTO live_predictions (
        prediction_time,
        rf_prediction,
        rf_probability,
        xgb_prediction,
        xgb_probability
    )
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (prediction_time) DO NOTHING;
    """

    cursor.execute(
        insert_query,
        (
            prediction_time,
            rf_prediction,
            rf_probability,
            xgb_prediction,
            xgb_probability
        )
    )

    if cursor.rowcount == 1:
        print("New prediction saved to PostgreSQL!")
    else:
        print("Prediction already exists. Duplicate not inserted.")

    connection.commit()

    cursor.close()
    connection.close()

    print("PostgreSQL connection closed.")


if __name__ == "__main__":
    print("Database module ready.")