#train.py
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
import optuna
import joblib

def load_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        for line in file:
            entry = json.loads(line)
            
            if "skip" in entry and entry["skip"] == True:
                continue
            
            features = []
            for category, values in entry.items():
                for key, value in values.items():
                    if isinstance(value, str) and value.isdigit():
                        value = int(value)
                    features.append(value)
            
            accept_rate = features.pop(17)  # accept_rate is at index 17
            
            data.append(features)
            labels.append(accept_rate)
    
    return np.array(data), np.array(labels)

def feature_engineering(X):
    # Original features
    df = pd.DataFrame(X, columns=[
        'ethnicity', 'gender', 'income_bracket', 'type_school', 'app_round', 'gpa', 'ap_ib_courses',
        'ap_ib_scores', 'test_score', 'location', 'state_status', 'legacy', 'intended_major',
        'first_gen', 'languages', 'special_talents', 'hooks', 'nat_int', 'reg', 'local',
        'volunteering', 'ent', 'intern', 'add', 'res', 'sports', 'work_exp', 'leadership',
        'community_impact', 'ec_years', 'int_awards', 'nat_awards', 'state_awards', 'local_awards', 'other_awards'
    ])
    
    # Interaction terms
    df['gpa_test_score'] = df['gpa'] * df['test_score']
    df['ap_ib_total'] = df['ap_ib_courses'] * df['ap_ib_scores']
    df['income_first_gen'] = df['income_bracket'] * df['first_gen']
    
    # Aggregated features
    df['total_awards'] = df['int_awards'] + df['nat_awards'] + df['state_awards'] + df['local_awards'] + df['other_awards']
    df['total_ecs'] = df['nat_int'] + df['reg'] + df['local'] + df['volunteering'] + df['ent'] + df['intern'] + df['add'] + df['res'] + df['sports'] + df['work_exp']
    
    # Polynomial features for important columns
    for col in ['gpa', 'test_score', 'ap_ib_total', 'total_awards', 'total_ecs']:
        df[f'{col}_squared'] = df[col] ** 2
    
    # Normalize ec_years by total_ecs
    df['avg_ec_years'] = df['ec_years'] / (df['total_ecs'] + 1)  # Add 1 to avoid division by zero
    
    # Encoding categorical variables
    df = pd.get_dummies(df, columns=['ethnicity', 'gender', 'type_school', 'location'])
    return df

def create_nn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  loss='mse', metrics=['mae'])
    return model

def objective(trial, X, y):
    model_type = trial.suggest_categorical('model_type', ['RF', 'XGB', 'LGBM'])
    
    if model_type == 'RF':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
        }
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    elif model_type == 'XGB':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
        }
        model = XGBRegressor(**params, random_state=42, n_jobs=-1)
    else:  # LGBM
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
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
    X, y = load_data('categorization/categorized.json')
    
    # Convert y to categorical for stratification
    y_cat = pd.cut(y, bins=[-np.inf, 0.5, 1.5, 2.5, np.inf], labels=[0, 1, 2, 3])
    
    X = feature_engineering(X)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    # Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y_cat)
    
    # Convert y back to continuous
    y_resampled = y_resampled.astype(float)
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_scaled, y_train), n_trials=100)
    
    best_params = study.best_params
    best_model_type = best_params.pop('model_type')
    
    if best_model_type == 'RF':
        best_model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
    elif best_model_type == 'XGB':
        best_model = XGBRegressor(**best_params, random_state=42, n_jobs=-1)
    else:  # LGBM
        best_model = LGBMRegressor(**best_params, random_state=42, n_jobs=-1)
    
    best_model.fit(X_train_scaled, y_train)
    y_pred = best_model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"Best Model: {best_model_type}")
    print(f"Best Parameters: {best_params}")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Save the best model
    joblib.dump(best_model, f'best_model_{best_model_type.lower()}.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    
    # Train Neural Network
    nn_model = create_nn_model(X_train_scaled.shape[1])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    nn_model.fit(X_train_scaled, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
    
    y_pred_nn = nn_model.predict(X_test_scaled).flatten()
    mse_nn = mean_squared_error(y_test, y_pred_nn)
    mae_nn = mean_absolute_error(y_test, y_pred_nn)
    
    print("\nNeural Network Results:")
    print(f"MSE: {mse_nn:.4f}")
    print(f"MAE: {mae_nn:.4f}")
    
    # Save Neural Network model
    nn_model.save('best_model_nn.keras')
    
    # Ensemble prediction
    y_pred_ensemble = (y_pred + y_pred_nn) / 2
    mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
    mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
    
    print("\nEnsemble Results:")
    print(f"MSE: {mse_ensemble:.4f}")
    print(f"MAE: {mae_ensemble:.4f}")

if __name__ == "__main__":
    main()