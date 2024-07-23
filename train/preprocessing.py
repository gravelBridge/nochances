import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import json

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
    
    # One-hot encode
    categorical_columns = ['ethnicity', 'gender', 'type_school', 'location', 
                           'app_round', 'state_status', 'legacy', 'first_gen', 
                           'special_talents']
    # Encoding categorical variables
    df = pd.get_dummies(df, columns=categorical_columns)
    return df

def preprocess_data(X, y=None, is_training=True):
    X = feature_engineering(X)
    
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    
    if is_training and y is not None:
        # Convert y to categorical for stratification
        y_cat = pd.cut(y, bins=[-np.inf, 0.5, 1.5, 2.5, np.inf], labels=[0, 1, 2, 3])
        
        # Apply SMOTE to balance the dataset
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y_cat)
        
        # Convert y back to continuous
        y_resampled = y_resampled.astype(float)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_resampled)
        
        return X_scaled, y_resampled, scaler
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler