import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.preprocessing import load_data, preprocess_data, feature_engineering

def get_feature_names():
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

    # Sort by importance and get top 20
    feature_importance_df = feature_importance_df.sort_values(
        "importance", ascending=False
    ).head(20)

    # Print top 20 most important features
    print("Top 20 Most Important Features:")
    print(feature_importance_df.to_string(index=False))

    # Set up the matplotlib figure
    plt.figure(figsize=(14, 12))
    
    # Create the plot
    sns.set(style="whitegrid", font_scale=1.3)
    ax = sns.barplot(x="importance", y="feature", data=feature_importance_df, orient="h")

    # Customize the plot
    plt.title("Top 20 Feature Importance", fontsize=24, fontweight='bold')
    plt.xlabel("Importance Score", fontsize=24, fontweight='bold')
    plt.ylabel("Features", fontsize=24, fontweight='bold')

    # Make y-axis labels (feature names) bold and larger
    plt.yticks(fontsize=24, fontweight='bold')

    # Adjust layout to reduce margins
    plt.tight_layout()
    
    # Create subfolder if it doesn't exist
    os.makedirs("info/feature_analysis", exist_ok=True)
    
    # Save the figure
    plt.savefig("info/feature_analysis/feature_importance.png", bbox_inches='tight', dpi=300)
    print("\nFeature importance plot saved as 'info/feature_analysis/feature_importance.png'")

    return feature_importance_df


def analyze_feature_values(X, y, feature_importance_df):
    xgb_model = joblib.load("models/best_model_xgb.joblib")
    y_pred = xgb_model.predict(X)
    feature_names = get_feature_names()
    summary_data = []

    for feature in feature_importance_df['feature']:
        feature_index = feature_names.index(feature)
        feature_values = X[:, feature_index]
        
        if np.issubdtype(feature_values.dtype, np.number):
            # Continuous feature
            bins = pd.cut(feature_values, bins=10)
            bin_means = pd.Series(y_pred).groupby(bins).mean()
            
            plt.figure(figsize=(10, 6))
            bin_means.plot(kind='line', marker='o')
            plt.title(f"Average Model Output vs {feature}")
            plt.xlabel(feature)
            plt.ylabel("Average Model Output")
            plt.tight_layout()
            plt.savefig(f"info/feature_analysis/{feature}_analysis.png", dpi=300)
            plt.close()

            lowest_output = bin_means.min()
            highest_output = bin_means.max()
            percent_change = (highest_output - lowest_output) / lowest_output * 100
            direction = "Increase" if bin_means.idxmax() > bin_means.idxmin() else "Decrease"

            summary_data.append({
                'Feature': feature,
                'Type': 'Continuous',
                'Lowest Output Value/Range': str(bin_means.idxmin()),
                'Highest Output Value/Range': str(bin_means.idxmax()),
                'Percent Change': percent_change,
                'Direction': direction
            })

        else:
            # Categorical feature
            unique_values = np.unique(feature_values)
            cat_means = [np.mean(y_pred[feature_values == val]) for val in unique_values]
            
            plt.figure(figsize=(10, 6))
            plt.bar(unique_values, cat_means)
            plt.title(f"Average Model Output vs {feature}")
            plt.xlabel(feature)
            plt.ylabel("Average Model Output")
            plt.tight_layout()
            plt.savefig(f"info/feature_analysis/{feature}_analysis.png", dpi=300)
            plt.close()

            lowest_output = min(cat_means)
            highest_output = max(cat_means)
            percent_change = (highest_output - lowest_output) / lowest_output * 100
            low_value = unique_values[np.argmin(cat_means)]
            high_value = unique_values[np.argmax(cat_means)]

            summary_data.append({
                'Feature': feature,
                'Type': 'Categorical',
                'Lowest Output Value/Range': low_value,
                'Highest Output Value/Range': high_value,
                'Percent Change': percent_change,
                'Direction': f"{low_value} to {high_value}"
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv("info/feature_analysis/feature_value_summary.csv", index=False)
    print("\nFeature value summary saved as 'info/feature_analysis/feature_value_summary.csv'")
    
    return summary_df

def plot_feature_impact(summary_df):
    # Sort features by the magnitude of their percent change impact
    summary_df['Abs Percent Change'] = summary_df['Percent Change'].abs()
    top_features = summary_df.nlargest(20, 'Abs Percent Change')

    # Create a horizontal bar plot
    plt.figure(figsize=(16, 14))
    
    # Use a diverging color palette
    colors = ['#d7191c' if direction == 'Decrease' else '#2c7bb6' for direction in top_features['Direction']]
    
    bars = plt.barh(top_features['Feature'], top_features['Percent Change'], color=colors)

    # Customize the plot
    plt.title('Top 20 Features: Impact on Model Output and Direction of Influence', fontsize=16, fontweight='bold')
    plt.xlabel('Percent Change in Model Output', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    # Add value labels and direction to the end of each bar
    for i, (feature, v, direction, low, high) in enumerate(zip(
        top_features['Feature'], top_features['Percent Change'], 
        top_features['Direction'], top_features['Lowest Output Value/Range'], 
        top_features['Highest Output Value/Range']
    )):
        if top_features['Type'].iloc[i] == 'Categorical':
            label = f"{low} → {high}" if direction == 'Increase' else f"{high} → {low}"
        else:
            label = "↑" if direction == 'Increase' else "↓"
        plt.text(v, i, f' {abs(v):.2f}% ({label})', va='center', fontweight='bold')

    # Add a legend
    red_patch = plt.Rectangle((0, 0), 1, 1, fc="#d7191c")
    blue_patch = plt.Rectangle((0, 0), 1, 1, fc="#2c7bb6")
    plt.legend([red_patch, blue_patch], 
               ['Lower values increase output', 'Higher values increase output'],
               loc='lower right')

    plt.tight_layout()
    plt.savefig("info/feature_analysis/top_20_feature_impact.png", dpi=300, bbox_inches='tight')
    print("\nTop 20 Feature Impact plot saved as 'info/feature_analysis/top_20_feature_impact.png'")

    # Create a table with detailed information
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.axis('off')
    table = ax.table(cellText=top_features[['Feature', 'Percent Change', 'Lowest Output Value/Range', 'Highest Output Value/Range']].values,
                     colLabels=['Feature', 'Percent Change', 'Lowest Output Value/Range', 'Highest Output Value/Range'],
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)
    plt.title('Top 20 Features - Detailed Information', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("info/feature_analysis/top_20_feature_impact_table.png", dpi=300, bbox_inches='tight')
    print("\nTop 20 Feature Impact table saved as 'info/feature_analysis/top_20_feature_impact_table.png'")

def main():
    # Load and preprocess the data
    X, y = load_data("categorization/categorized.json")
    X_preprocessed, _, _ = preprocess_data(X, y, is_training=True)

    # Analyze feature importance
    feature_importance_df = analyze_feature_importance()

    # Analyze feature values
    summary_df = analyze_feature_values(X_preprocessed, y, feature_importance_df)

    # Plot feature impact
    plot_feature_impact(summary_df)

if __name__ == "__main__":
    main()