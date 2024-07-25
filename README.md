# NoChances: College Admission Probability Predictor

## Predictive Modeling of College Admission Probabilities Using Machine Learning and Natural Language Processing on Unstructured Online Data Parsed by GPT-4o

## Overview

NoChances is an advanced machine learning project that predicts college admission probabilities using natural language processing on unstructured online data, augmented by Large Language Models (LLMs). This project analyzes Reddit posts from r/collegeresults to extract relevant information and build a predictive model for college admissions.

## Features

- Data scraping from Reddit using PRAW
- Data extraction and categorization using GPT-4
- Comprehensive feature engineering
- Multiple machine learning models:
  - XGBoost
  - Random Forest
  - LightGBM
  - Neural Network
- Ensemble prediction combining multiple models
- Principal Component Analysis (PCA) for feature importance visualization
- Inference module for real-time predictions

## Model Performance

The project uses an ensemble of models to achieve high accuracy. The current performance metrics are:

- XGBoost MSE (on entire dataset): 0.0835
- Neural Network MSE (on entire dataset): 0.2337
- Ensemble MSE (on entire dataset): 0.1136

## Contributing

Contributions to NoChances are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature-branch-name`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch-name`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- John Tian and Yourui Shao, co-first authors
- Reddit community r/collegeresults for providing valuable data
- OpenAI for GPT-4o API used in data categorization
- Contributors and maintainers of the various machine learning libraries used in this project
