import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from preprocessing import load_data, preprocess_data


def create_nn_model(input_shape):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_shape,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(
                256,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(
                128,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(
                64,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(
                32,
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

    # Train Neural Network
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
        verbose=1
    )

    y_pred_nn = nn_model.predict(X_test).flatten()
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    mae_nn = mean_absolute_error(y_test, y_pred_nn)

    print("\nNeural Network Results:")
    print(f"MSE: {mse_nn:.4f}")
    print(f"MAE: {mae_nn:.4f}")

    # Save Neural Network model
    nn_model.save("models/best_model_nn.keras")

    # Save scaler
    joblib.dump(scaler, "models/scaler.joblib")


if __name__ == "__main__":
    main()