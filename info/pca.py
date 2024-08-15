import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams.update({'font.size': 14})  # Increase the default font size

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from the train directory
from train.preprocessing import load_data, preprocess_data, feature_engineering

def get_numeric_features(X):
    # Get original feature names
    original_features = [
        "ethnicity", "gender", "income_bracket", "type_school", "app_round",
        "gpa", "ap_ib_courses", "ap_ib_scores", "test_score", "location",
        "state_status", "legacy", "intended_major", "major_alignment",
        "first_gen", "languages", "special_talents", "hooks",
        "nat_int", "reg", "local", "volunteering", "ent", "intern", "add",
        "res", "sports", "work_exp", "leadership", "community_impact",
        "ec_years", "int_awards", "nat_awards", "state_awards",
        "local_awards", "other_awards"
    ]
    
    # List of categorical columns
    categorical_columns = ["ethnicity", "gender", "type_school", "location", "app_round",
                           "state_status", "legacy", "first_gen", "special_talents"]
    
    # Get indices of non-categorical columns
    numeric_indices = [i for i, feature in enumerate(original_features) if feature not in categorical_columns]
    
    # Get all column indices
    all_indices = list(range(X.shape[1]))
    
    # Keep only the original numeric columns and any additional engineered features
    numeric_columns = [col for col in all_indices if col in numeric_indices or col >= len(original_features)]
    
    return numeric_columns

def perform_pca(X, n_components=None):
    # Select only numeric features
    numeric_columns = get_numeric_features(X)
    X_numeric = X.iloc[:, numeric_columns]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)

    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return pca, X_pca, numeric_columns

def plot_explained_variance(pca):
    plt.figure(figsize=(12, 7))
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, "bo-", markersize=8)
    plt.xlabel("Number of Components", fontsize=16)
    plt.ylabel("Cumulative Explained Variance Ratio", fontsize=16)
    plt.title("Explained Variance Ratio vs. Number of Components", fontsize=18, pad=20)
    plt.grid(True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("info/pca_explained_variance.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_pca_components(X_pca, y, n_components=2):
    plt.figure(figsize=(14, 12))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.6, s=50)
    cbar = plt.colorbar(scatter, label="Accept Rate Category")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Accept Rate Category", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=16)
    plt.ylabel("Principal Component 2", fontsize=16)
    plt.title("PCA of College Admissions Data (Ordinal Features Only)", fontsize=18, pad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("info/pca_components.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(pca, feature_names):
    components = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=feature_names,
    )
    plt.figure(figsize=(22, 12))
    sns.heatmap(components, cmap="coolwarm", annot=False, cbar_kws={'label': 'Component Coefficient'})
    plt.title("Feature Importance in Principal Components (Ordinal Features Only)", fontsize=18, pad=20)
    plt.xlabel("Principal Components", fontsize=16)
    plt.ylabel("Features", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14, rotation=0)
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=14)
    cbar.set_ylabel("Component Coefficient", fontsize=16, rotation=270, labelpad=20)
    plt.tight_layout()
    plt.savefig("info/pca_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load and preprocess the data
    X, y = load_data("categorization/categorized.json")
    X_preprocessed, y_preprocessed, _ = preprocess_data(X, y, is_training=True)

    # Convert X_preprocessed to DataFrame if it's not already
    if not isinstance(X_preprocessed, pd.DataFrame):
        X_preprocessed = pd.DataFrame(X_preprocessed)

    # Perform PCA
    pca, X_pca, numeric_features = perform_pca(X_preprocessed)

    # Plot explained variance
    plot_explained_variance(pca)

    # Plot first two PCA components
    plot_pca_components(X_pca, y_preprocessed)

    # Plot feature importance
    plot_feature_importance(pca, numeric_features)

    # Print the explained variance ratio
    print("Explained Variance Ratio:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {ratio:.4f}")

    # Print the cumulative explained variance ratio
    print("\nCumulative Explained Variance Ratio:")
    cumulative_ratio = np.cumsum(pca.explained_variance_ratio_)
    for i, ratio in enumerate(cumulative_ratio):
        print(f"PC1-PC{i+1}: {ratio:.4f}")

    # Determine number of components for 95% variance explained
    n_components_95 = np.argmax(cumulative_ratio >= 0.95) + 1
    print(f"\nNumber of components explaining 95% of variance: {n_components_95}")

if __name__ == "__main__":
    main()