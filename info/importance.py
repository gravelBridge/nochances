import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.preprocessing import load_data, preprocess_data, feature_engineering


def get_feature_names():
    # This function returns a list of feature names based on the feature engineering process
    X, _ = load_data("categorization/categorized.json")
    df = feature_engineering(X)
    return df.columns.tolist()


def analyze_feature_importance():
    # Load the trained XGBoost model
    xgb_model = joblib.load("models/best_model_xgb.joblib")

    # Get feature names
    feature_names = get_feature_names()

    # Get feature importance scores
    importance_scores = xgb_model.feature_importances_

    # Create a dataframe with feature names and importance scores
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance_scores}
    )

    # Sort by importance
    feature_importance_df = feature_importance_df.sort_values(
        "importance", ascending=False
    )

    # Print top 20 most important features
    print("Top 20 Most Important Features:")
    print(feature_importance_df.head(20).to_string(index=False))

    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.bar(
        feature_importance_df["feature"][:20], feature_importance_df["importance"][:20]
    )
    plt.xticks(rotation=90)
    plt.title("Top 20 Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("info/feature_importance.png")
    print("\nFeature importance plot saved as 'info/feature_importance.png'")


if __name__ == "__main__":
    analyze_feature_importance()
