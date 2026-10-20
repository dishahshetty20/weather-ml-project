-- Safe additive schema: creates new regression tables only.
-- Existing historical_weather, live_predictions, and
-- live_prediction_evaluations tables are not modified.

CREATE TABLE IF NOT EXISTS live_regression_predictions (
    id BIGSERIAL PRIMARY KEY,
    observation_time TIMESTAMP NOT NULL,
    prediction_for TIMESTAMP NOT NULL,
    target TEXT NOT NULL,
    method TEXT NOT NULL,
    predicted_value DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_live_regression_prediction
        UNIQUE (prediction_for, target, method)
);

CREATE TABLE IF NOT EXISTS live_regression_evaluations (
    id BIGSERIAL PRIMARY KEY,
    prediction_for TIMESTAMP NOT NULL,
    target TEXT NOT NULL,
    method TEXT NOT NULL,
    predicted_value DOUBLE PRECISION NOT NULL,
    actual_value DOUBLE PRECISION NOT NULL,
    error DOUBLE PRECISION NOT NULL,
    absolute_error DOUBLE PRECISION NOT NULL,
    evaluated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_live_regression_evaluation
        UNIQUE (prediction_for, target, method)
);
