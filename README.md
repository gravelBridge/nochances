# Predicting College Admission Results with Machine Learning on Unstructured Online Data

This repository accompanies the paper [*"Predicting College Admission Results with Machine Learning on Unstructured Online Data"*](https://drive.google.com/file/d/1POmM-Nsp_GbBwEzh3vb2-_D-_eP7Ibeh/view?usp=sharing). It leverages advanced language models (GPT-4o) and traditional machine learning techniques (XGBoost, Neural Networks, Random Forests) to predict US college admission outcomes from user-submitted, unstructured application data posted on the r/collegeresults subreddit. By bridging the gap between raw, unstructured text and structured prediction features, this project aims to provide greater transparency and data-driven insights into the opaque college admissions process.

## Table of Contents
- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Dataset](#dataset)
- [Approach](#approach)
  - [Data Collection](#data-collection)
  - [Feature Extraction with GPT-4o](#feature-extraction-with-gpt-4o)
  - [Preprocessing and Balancing](#preprocessing-and-balancing)
  - [Modeling Strategies](#modeling-strategies)
- [Models and Results](#models-and-results)
  - [Method 1: Tier-Based Prediction](#method-1-tier-based-prediction)
  - [Method 2: Institution-Specific Prediction](#method-2-institution-specific-prediction)
  - [Performance Metrics](#performance-metrics)
- [Analysis and Interpretations](#analysis-and-interpretations)
  - [Feature Importance](#feature-importance)
  - [Principal Component Analysis](#principal-component-analysis)
- [Limitations and Future Work](#limitations-and-future-work)
- [Usage](#usage)
  - [Prerequisites](#prerequisites)
  - [Running the Code](#running-the-code)
- [Web Interface](#web-interface)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Overview
College admissions in the United States are often characterized by complexity and opacity. High-achieving students must navigate processes influenced by academic metrics, extracurricular involvements, awards, socioeconomic background, and other factors. Our research leverages a novel data source—posts on the r/collegeresults subreddit—to train models that predict admissions results. We show that advanced NLP techniques can extract structured features from unstructured personal statements and application summaries, ultimately producing a model that offers valuable predictions and insights into the admissions process.

## Key Contributions
1. **Novel Data Source**: Leveraging over 4,000 user-submitted posts from r/collegeresults, we extract application features and outcomes not readily available in traditional datasets.
2. **GPT-4o Feature Extraction**: We use GPT-4o to transform unstructured text data into structured features such as GPA tiers, extracurricular intensities, award levels, and demographic indicators.
3. **Two Modeling Approaches**:
   - **Method 1 (Tier-Based)**: Predict the *tier* of the most selective school a student could be admitted to, achieving ~91.66% accuracy.
   - **Method 2 (Institution-Specific)**: Predict acceptance probabilities for individual schools, comparing traditional tokenization methods with GPT-4o-based descriptive features.
4. **Public Accessibility**: Our tier-based model is deployed at [acceptifyai.com](http://acceptifyai.com), allowing students to receive predictions based on their profile.
5. **Insights into Factors**: Feature importance analysis reveals factors like GPA, test scores, and community impact as top predictors. This can help identify biases and better understand admissions.

## Dataset
Our dataset comprises:
- **r/collegeresults Subreddit Posts**: Collected from 2020 to 2024, each post includes applicant demographics, academic metrics, test scores, extracurriculars, awards, and a list of colleges with results.
- **Data Augmentation**: Synthetic "below-average" profiles are added to counteract self-selection bias from subreddit posts, creating a more balanced training set.
- **External Data**: Institutional acceptance rates and major popularity data are incorporated to refine predictions in the institution-specific approach.

## Approach

### Data Collection
- **Reddit API and Pushshift Archives**: We gathered 4,127 posts, filtering out removed or incomplete data.
- **Unstructured to Structured**: Each post’s text-based content is passed to GPT-4o, which outputs a structured JSON with standardized categorical and ordinal features.

### Feature Extraction with GPT-4o
- GPT-4o is prompted with detailed instructions and a strict JSON schema.
- Extracted features include GPA tiers, test score ranges, number of AP/IB courses, extracurricular categories, and award levels.
- Textual mentions of activities and achievements are converted into numerical features.

### Preprocessing and Balancing
- **SMOTE**: Applied to handle class imbalance, especially in demographic features.
- **Data Augmentation**: Artificially created low-statistic profiles to broaden the model’s applicability.
- **Feature Engineering**: Interaction and polynomial terms are added to improve model performance.

### Modeling Strategies
**Method 1 (Tier Prediction)**:
- Predict the selectivity tier of the most selective admitted institution.
- Ensemble of XGBoost and a Neural Network model.

**Method 2 (Institution-Specific Prediction)**:
- Predict the probability of acceptance at a particular institution.
- Compare token-based text vectorization with GPT-4o-generated descriptive features and a hybrid approach (IS-T-D).

## Models and Results
### Method 1: Tier-Based Prediction
- **Input**: A 63-feature vector (after one-hot encoding and feature engineering).
- **Output**: An integer category representing the most selective institution tier the student can get into.
- **Performance**: 
  - Accuracy: 91.66%
  - AUC-ROC: 0.9298

### Method 2: Institution-Specific Prediction
- **Input**: Expanded dataset of individual school applications (~22k data points).
- **Approaches**:
  - IS-D (Descriptive only)
  - IS-T (Token-based only)
  - IS-T-D (Hybrid)
- **Results**:
  - Tokenization-based features slightly outperform GPT-4o descriptive features (85.1% vs. 84.3% accuracy).
  - Hybrid does not significantly improve performance beyond tokenization alone.

### Performance Metrics
- **Method 1**: Achieved high accuracy, precision, recall, and AUC-ROC, indicating reliable tier classification.
- **Method 2**: Achieved ~85% accuracy in predicting acceptance outcomes for specific institutions.

## Analysis and Interpretations
### Feature Importance
- High-impact features include GPA, test score tiers, community impact, and certain "hooks".
- Demographic factors show influence but are overshadowed by strong academic metrics and significant extracurricular or research involvements.

### Principal Component Analysis
- PCA reveals a complex, multi-dimensional data space. Many principal components are needed to explain the variance, highlighting the complexity and multifactorial nature of admissions decisions.

## Limitations and Future Work
- **Data Source Bias**: r/collegeresults skews towards competitive applicants and top-tier institutions.
- **Missing Factors**: Essays, recommendation letters, and interviews—critical in real admissions—are not included.
- **Future Directions**:
  - Incorporating more representative data sources.
  - Employing advanced NLP for essay and recommendation letter analysis.
  - Enhancing embeddings or extraction methods to better capture nuanced factors.

## Usage

### Prerequisites
- Python 3.8+
- Install dependencies from `requirements.txt`

### Running the Code
1. **Data Collection**: Use the scripts in `scraping` directory.
2. **Categorization**: Run GPT-4o-based feature extraction using the `categorization` directory.
3. **Training**: Use `train/train.py` to run model training and hyperparameter tuning.
4. **Testing**: Evaluate performance with `train/test_models.py`.

Example:
```bash
python3 train/train.py
python3 train/test_models.py
```

## Web Interface
A Flask-based web application is included in `webapp`. After training and saving models:
```bash
cd webapp
flask run
```
Visit `http://localhost:5000` to interact with the predictor. A live demo is available at [acceptifyai.com](acceptifyai.com).

## Acknowledgments

- **Co-First Authors**: John Tian and Yourui Shao
- **Support**: Dr. Hugh Tad Blair (UCLA) for guidance and partial funding
- **Data**: r/collegeresults subreddit community
- **Tools**: GPT-4o, XGBoost, TensorFlow, scikit-learn, Optuna

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
