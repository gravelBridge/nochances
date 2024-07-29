import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
import joblib
from preprocessing import load_data, preprocess_data


def create_nn_model(input_shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_shape,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(
                1000,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(
                1000,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                1000,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(
                1000,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(
                1000,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model


def objective(trial, X, y):
    model_type = trial.suggest_categorical("model_type", ["RF", "XGB", "LGBM"])

    if model_type == "RF":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    elif model_type == "XGB":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = XGBRegressor(**params, random_state=42, n_jobs=-1)
    else:  # LGBM
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        model = LGBMRegressor(**params, random_state=42, n_jobs=-1)

    scores = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        score = mean_absolute_error(y_val, y_pred)
        scores.append(score)

    return np.mean(scores)


def main():
    X, y = load_data("categorization/categorized.json")

    X_preprocessed, y_preprocessed, scaler = preprocess_data(X, y, is_training=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed,
        y_preprocessed,
        test_size=0.2,
        random_state=42,
        stratify=y_preprocessed,
    )

    nn_model = create_nn_model(X_train.shape[1])
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )
    nn_model.fit(
        X_train,
        y_train,
        epochs=200,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1,
    )

    y_pred_nn = nn_model.predict(X_test).flatten()
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    mae_nn = mean_absolute_error(y_test, y_pred_nn)

    # Save Neural Network model
    nn_model.save("models/best_model_nn.keras")

    # Hyperparameter tuning
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=200)

    best_params = study.best_params
    best_model_type = best_params.pop("model_type")

    if best_model_type == "RF":
        best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    elif best_model_type == "XGB":
        best_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    else:  # LGBM
        best_model = LGBMRegressor(**best_params, random_state=42, n_jobs=-1)

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Best Model: {best_model_type}")
    print(f"Best Parameters: {best_params}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Save the best model and scaler
    joblib.dump(best_model, f"models/best_model_{best_model_type.lower()}.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    # Ensemble prediction
    y_pred_ensemble = (y_pred + y_pred_nn) / 2
    mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
    mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)

    print("\nNeural Network Results:")
    print(f"MSE: {mse_nn:.4f}")
    print(f"MAE: {mae_nn:.4f}")

    print("\nEnsemble Results:")
    print(f"MSE: {mse_ensemble:.4f}")
    print(f"MAE: {mae_ensemble:.4f}")


if __name__ == "__main__":
    main()