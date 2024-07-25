import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from preprocessing import load_data, preprocess_data


def load_models():
    xgb_model = joblib.load("models/best_model_xgb.joblib")
    nn_model = tf.keras.models.load_model("models/best_model_nn.keras")
    scaler = joblib.load("models/scaler.joblib")
    return xgb_model, nn_model, scaler


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    if isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
        y_pred = y_pred.flatten()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n{model_name} Results:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")

    return y_pred


def main():
    # Load the models and scaler first
    xgb_model, nn_model, scaler = load_models()

    # Load and preprocess the data
    X, y = load_data("categorization/categorized.json")
    X_preprocessed = preprocess_data(X, y, is_training=False, scaler=scaler)

    # Evaluate XGBoost model
    xgb_pred = evaluate_model(xgb_model, X_preprocessed, y, "XGBoost")

    # Evaluate Neural Network model
    nn_pred = evaluate_model(nn_model, X_preprocessed, y, "Neural Network")

    # Evaluate Ensemble
    ensemble_pred = (xgb_pred + nn_pred) / 2
    mse_ensemble = mean_squared_error(y, ensemble_pred)
    mae_ensemble = mean_absolute_error(y, ensemble_pred)

    print("\nEnsemble Results:")
    print(f"MSE: {mse_ensemble:.4f}")
    print(f"MAE: {mae_ensemble:.4f}")


if __name__ == "__main__":
    main()
