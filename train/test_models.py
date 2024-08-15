import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import (mean_squared_error, mean_absolute_error, accuracy_score,
 precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import label_binarize
from preprocessing import load_data, preprocess_data

def load_models():
    xgb_model = joblib.load("models/best_model_xgb.joblib")
    nn_model = tf.keras.models.load_model("models/best_model_nn.keras")
    scalers = joblib.load("models/scalers.joblib")
    return xgb_model, nn_model, scalers

def evaluate_model(model, X_test, y_test, model_name):
    # Check if the model is a callable (function) or has a predict method
    if callable(model):
        y_pred = model(X_test)
    elif hasattr(model, 'predict'):
        y_pred = model.predict(X_test)
    else:
        raise ValueError("Model must be either callable or have a 'predict' method")

    # Ensure predictions and true labels are 1D arrays
    y_pred = np.array(y_pred).ravel()
    y_test = np.array(y_test).ravel()

    # Round predictions to nearest integer for classification metrics
    y_pred_rounded = np.round(y_pred).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_rounded)
    precision = precision_score(y_test, y_pred_rounded, average='weighted')
    recall = recall_score(y_test, y_pred_rounded, average='weighted')

    # For AUC-ROC, we need to binarize the true labels and predictions
    n_classes = len(np.unique(y_test))
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    y_pred_bin = label_binarize(y_pred_rounded, classes=range(n_classes))

    try:
        auc_roc = roc_auc_score(y_test_bin, y_pred_bin, average='weighted', multi_class='ovr')
    except ValueError:
        auc_roc = None  # In case of errors (e.g., a class has no samples)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    if auc_roc is not None:
        print(f"AUC-ROC: {auc_roc:.4f}")
    else:
        print("AUC-ROC: Not calculated (possibly due to sample issues)")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")

    return y_pred

def main():
    # Load the models and scalers first
    xgb_model, nn_model, scalers = load_models()

    # Load and preprocess the data
    X, y = load_data("categorization/categorized.json")
    X_preprocessed = preprocess_data(X, y, is_training=False, scalers=scalers)

    # Evaluate XGBoost model
    xgb_pred = evaluate_model(xgb_model, X_preprocessed, y, "XGBoost")

    # Evaluate Neural Network model
    nn_pred = evaluate_model(nn_model, X_preprocessed, y, "Neural Network")

    # Evaluate Ensemble
    ensemble_pred = (xgb_pred + nn_pred) / 2
    evaluate_model(lambda x: ensemble_pred, X_preprocessed, y, "Ensemble")

if __name__ == "__main__":
    main()