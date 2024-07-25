import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from the train directory
from train.preprocessing import load_data, preprocess_data


def perform_pca(X, n_components=None):
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    return pca, X_pca


def plot_explained_variance(pca):
    plt.figure(figsize=(10, 6))
    cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(
        range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, "bo-"
    )
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.title("Explained Variance Ratio vs. Number of Components")
    plt.grid(True)
    plt.savefig("info/pca_explained_variance.png")
    plt.close()


def plot_pca_components(X_pca, y, n_components=2):
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.6)
    plt.colorbar(scatter, label="Accept Rate Category")
    plt.xlabel(f"Principal Component 1")
    plt.ylabel(f"Principal Component 2")
    plt.title(f"PCA of College Admissions Data")
    plt.savefig("info/pca_components.png")
    plt.close()


def plot_feature_importance(pca, feature_names):
    components = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=feature_names,
    )
    plt.figure(figsize=(20, 10))
    sns.heatmap(components, cmap="coolwarm", annot=False)
    plt.title("Feature Importance in Principal Components")
    plt.savefig("info/pca_feature_importance.png")
    plt.close()


def main():
    # Load and preprocess the data
    X, y = load_data("categorization/categorized.json")
    X_preprocessed, y_preprocessed, _ = preprocess_data(X, y, is_training=True)

    # Get feature names
    feature_names = (
        X_preprocessed.columns
        if isinstance(X_preprocessed, pd.DataFrame)
        else [f"Feature_{i}" for i in range(X_preprocessed.shape[1])]
    )

    # Perform PCA
    pca, X_pca = perform_pca(X_preprocessed)

    # Plot explained variance
    plot_explained_variance(pca)

    # Plot first two PCA components
    plot_pca_components(X_pca, y_preprocessed)

    # Plot feature importance
    plot_feature_importance(pca, feature_names)

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
