import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import json


def load_data(file_path, inference=False):
    data = []
    labels = []
    with open(file_path, "r") as file:
        for line in file:
            entry = json.loads(line)

            if "skip" in entry and entry["skip"] == True:
                continue

            features = []
            accept_rate = None
            for category, values in entry.items():
                if category == "basic_info":
                    accept_rate = values.get("accept_rate")
                for key, value in values.items():
                    if key != "accept_rate":
                        if isinstance(value, str) and value.isdigit():
                            value = int(value)
                        features.append(value)

            if not inference:
                if accept_rate is None:
                    raise ValueError("accept_rate not found in the data")
                labels.append(int(accept_rate))

            data.append(features)

    return np.array(data), np.array(labels) if not inference else None


def feature_engineering(X):
    # Convert X to DataFrame if it's not already
    df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
    
    # Define column names if they're not already set
    if df.columns.dtype == 'int64':
        df.columns = [
            "ethnicity", "gender", "income_bracket", "type_school", "app_round",
            "gpa", "ap_ib_courses", "ap_ib_scores", "test_score", "location",
            "state_status", "legacy", "intended_major", "major_alignment",
            "first_gen", "languages", "special_talents", "hooks",
            "nat_int", "reg", "local", "volunteering", "ent", "intern", "add",
            "res", "sports", "work_exp", "leadership", "community_impact",
            "ec_years", "int_awards", "nat_awards", "state_awards",
            "local_awards", "other_awards"
        ]

    # Apply StandardScaler to original numerical features
    original_num_features = ["gpa", "ap_ib_courses", "ap_ib_scores", "test_score", 
                             "income_bracket", "languages", "hooks", "nat_int", 
                             "reg", "local", "volunteering", "ent", "intern", 
                             "add", "res", "sports", "work_exp", "leadership", 
                             "ec_years", "int_awards", "nat_awards", "state_awards", 
                             "local_awards", "other_awards"]
    
    scaler_original = StandardScaler()
    df[original_num_features] = scaler_original.fit_transform(df[original_num_features])

    # Create engineered features using normalized original features
    df["gpa_test_score"] = df["gpa"] * df["test_score"]
    df["ap_ib_total"] = df["ap_ib_courses"] * df["ap_ib_scores"]
    df["income_first_gen"] = df["income_bracket"] * df["first_gen"]

    # Create polynomial features
    for col in ["gpa", "test_score", "ap_ib_total"]:
        df[f"{col}_squared"] = df[col] ** 2

    # Normalize ec_years by total extracurricular activities
    total_ecs = df["nat_int"] + df["reg"] + df["local"] + df["volunteering"] + df["ent"] + df["intern"] + df["add"] + df["res"] + df["sports"] + df["work_exp"]
    df["avg_ec_years"] = df["ec_years"] / (total_ecs + 1)  # Add 1 to avoid division by zero

    # Apply MinMaxScaler to engineered features
    engineered_features = ["gpa_test_score", "ap_ib_total", "income_first_gen", 
                           "gpa_squared", "test_score_squared", "ap_ib_total_squared", 
                           "avg_ec_years"]
    
    scaler_engineered = MinMaxScaler()
    df[engineered_features] = scaler_engineered.fit_transform(df[engineered_features])

    # One-hot encode categorical features
    categorical_columns = ["ethnicity", "gender", "type_school", "location", "app_round", 
                           "state_status", "legacy", "first_gen", "special_talents"]
    
    for col in categorical_columns:
        df = pd.get_dummies(df, columns=[col], prefix=col)

    return df

def preprocess_data(X, y=None, is_training=True, scaler=None):
    X = feature_engineering(X)

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    if is_training:
        if y is not None:
            # Convert y to categorical for stratification
            y_cat = pd.cut(y, bins=[-np.inf, 0.5, 1.5, 2.5, np.inf], labels=[0, 1, 2, 3])

            # Apply SMOTE to balance the dataset
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y_cat)

            # Convert y back to continuous
            y_resampled = y_resampled.astype(float)

            return X_resampled, y_resampled, None  # Return None instead of scaler
        else:
            return X, None  # Return None instead of scaler
    else:
        return X