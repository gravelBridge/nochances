from openai import OpenAI
import json
from dotenv import load_dotenv
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

load_dotenv()
client = OpenAI()

with open('categorization/prompt.txt', 'r') as f:
    prompt = f.read()

with open('train/inference/custom_input.txt', 'r') as f:
    flair = f.readline().strip()
    post = f.read()

JSON_SCHEMAS = {
    "basic_info": {
        "ethnicity": "Integer: 0 = Underrepresented Minority in College (Black, Hispanic, Native American, Pacific Islander), 1 = Not Underrepresented Minority in College (White, Asian, Other)",
        "gender": "Integer: 0 = Male, 1 = Female, 2 = Other/Non-binary",
        "income_bracket": "Integer: Family annual income pre-tax. 0 = $0-30k, 1 = $30k-60k, 2 = $60k-100k, 3 = $100k-200k, 4 = $200k+. Estimate if not specified.",
        "type_school": "Integer: Most selective accepted school type. 0 = STEM, 1 = Liberal Arts, 2 = Art/Design, 3 = Music Conservatory, 4 = Other specialized",
        "app_round": "Integer: For most selective accepted school. 0 = Early Decision/Action, 1 = Regular Decision",
        "gpa": "Integer: Unweighted 4.0 scale. Estimate conversion to UW 4.0 scale if not specified. 0 = Below 2.5, 1 = 2.5-2.99, 2 = 3.0-3.49, 3 = 3.5-3.79, 4 = 3.8-4.0",
        "ap-ib-courses": "Integer: Total number of AP and IB courses taken (exclude honors)",
        "ap-ib-scores": "Integer: 0 = No scores, 1 = Avg below 3(AP)/4(IB), 2 = Avg 3-3.9(AP)/4-4.9(IB), 3 = Avg 4-4.9(AP)/5-5.9(IB), 4 = Avg 5(AP)/6-7(IB)",
        "test-score": "Integer: 0 = No score/below 1000 SAT/20 ACT, 1 = 1000-1190 SAT/20-23 ACT, 2 = 1200-1390 SAT/24-29 ACT, 3 = 1400-1490 SAT/30-33 ACT, 4 = 1500+ SAT/34+ ACT",
        "location": "Integer: 0 = Rural (<10k), 1 = Suburban (10k-100k), 2 = Urban (>100k)",
        "state-status": "Integer: For most selective accepted school. 0 = In Same State, 1 = Not In Same State",
        "legacy": "Integer: For most selective accepted school. 0 = Legacy, 1 = Non-Legacy",
        "intended_major": "Integer 1-10: Competitiveness of major at most selective accepted school. 1 = Least competitive (e.g., Gender Studies at MIT), 10 = Most competitive (e.g., CS at MIT)",
        "first_gen": "Integer: 0 = Not first-generation college student, 1 = First-generation college student. Default to 0 if not specified",
        "languages": "Integer: Number of languages proficient in (including native). Default to 1 if not specified",
        "special_talents": "Integer: 0 = None, 1 = School/local (e.g., lead in school play), 2 = Regional (e.g., first chair in regional orchestra), 3 = National (e.g., national chess champion), 4 = International (e.g., junior Olympian). Default to 0 if not specified",
        "hooks": "Integer: Number of significant 'hooks' (e.g., recruited athlete, development case, child of faculty). Example: 2 for a recruited athlete whose parent is a professor",
        "accept_rate": "Integer: Selectivity of most selective accepted school. 0 = <5% (e.g., Harvard, MIT), 1 = 5-15% (e.g., UCLA, Cornell), 2 = 15-40% (e.g., UC Davis, Boston University), 3 = >40% (e.g., UC Santa Cruz, Arizona State) or Open Admission"
    },
    "ecs": {
        "nat-int": "Integer: Number of National/International Activities in notable position. Example: 1 for being president of a national youth organization",
        "reg": "Integer: Number of State/Regional Activities in notable position. Example: 2 for being captain of a regional debate team and organizing a state-wide science fair",
        "local": "Integer: Number of Local/School Activities/Clubs in notable position. Example: 3 for being president of school chess club, editor of school newspaper, and lead in school play",
        "volunteering": "Integer: Number of Volunteering Activities",
        "ent": "Integer: Number of Entrepreneurial ventures (founded/co-founded)",
        "intern": "Integer: Number of internships",
        "add": "Integer: Number of classes taken outside high school curriculum",
        "res": "Integer: Number of research papers completed",
        "sports": "Integer: Number of sports teams (e.g., JV, Varsity)",
        "work_exp": "Integer: Number of part-time jobs or significant family responsibilities",
        "leadership": "Integer: Total number of leadership positions across all activities",
        "community_impact": "Integer: 0 = None, 1 = Minimal (e.g., occasional volunteering), 2 = Moderate (e.g., regular volunteering), 3 = Significant (e.g., leading a community project), 4 = Exceptional with measurable impact (e.g., founding a successful non-profit)",
        "ec_years": "Integer: Average years of involvement across significant extracurriculars. Example: 3 if involved in most activities for 3 years of high school"
    },
    "awards": {
        "int": "Integer: Number of International Awards (1st-10th place or equivalent). Example: 1 for a bronze medal in International Math Olympiad",
        "nat": "Integer: Number of National Awards (1st-10th place or equivalent). Example: 2 for a 2nd place in National Science Fair and honorable mention in National Merit Scholarship",
        "state": "Integer: Number of State/Regional Awards (1st-10th place or equivalent). Example: 3 for winning state debate championship, 3rd in regional science fair, and 5th in state math competition",
        "local": "Integer: Number of Local/School Awards (1st-3rd place or equivalent). Example: 4 for various 1st-3rd place finishes in school competitions",
        "other": "Integer: Number of Honorable Mentions or Participation Awards (any level, no duplicates). Example: 2 for participation in Model UN and honorable mention in a national essay contest"
    }
}

json_string = json.dumps(JSON_SCHEMAS, indent=2)

def parse_and_append(json_string, filename):
    data = json.loads(json_string)
    with open(filename, 'w') as file:
        json.dump(data, file)
        file.write('\n')

def process_post():
    retries = 0
    while retries < 10:
        try:
            with open('train/inference/custom_input.txt', 'r') as f:
                flair = f.readline().strip()
                post = f.read()

            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Note:\nThe post is sourced from a Reddit forum. If it lacks sufficient information to make a determination, contains clearly false information, or is obviously a joke, output only {{\"skip\": true}}\n\nCRITICAL:\n\nYou MUST output a result for EVERY field even if not stated in the post.\nIf some piece of information is not stated in the post, make your best guess. But, if there is too much missing information, only output {{\"skip\": true}}\nDo not include any dialogue, only valid JSON.\n\nYou must only output valid JSON in this format:\n{json.dumps(JSON_SCHEMAS, indent=2)}\n\nHowever, if the post is a \"troll\" post, obviously fake, or contains too much missing information for the post to be accurately categorized/analyzed, only output {{\"skip\": true}}\n\nHere is the Reddit Post:\n{flair}\n{post}"}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = completion.choices[0].message.content
            
            parse_and_append(data, 'train/inference/gpt-4o_output.json')
            return
        except Exception as e:
            print(f"Error processing post: {e}")
            retries += 1
            if retries < 10:
                print(f"Retrying in 5 seconds... (Attempt {retries + 1}/10)")
                time.sleep(5)
            else:
                print(f"Failed to process post after 10 attempts.")

process_post()

def load_data(file_path):
    with open(file_path, 'r') as file:
        entry = json.loads(file.readline())
    
    features = []
    for category, values in entry.items():
        for key, value in values.items():
            if isinstance(value, str) and value.isdigit():
                value = int(value)
            features.append(value)
    
    accept_rate = None
    if 'accept_rate' in entry['basic_info']:
        accept_rate = features.pop(17)  # accept_rate is at index 17
    
    return np.array([features]), accept_rate

def feature_engineering(X):
    df = pd.DataFrame(X, columns=[
        'ethnicity', 'gender', 'income_bracket', 'type_school', 'app_round', 'gpa', 'ap_ib_courses',
        'ap_ib_scores', 'test_score', 'location', 'state_status', 'legacy', 'intended_major',
        'first_gen', 'languages', 'special_talents', 'hooks', 'nat_int', 'reg', 'local',
        'volunteering', 'ent', 'intern', 'add', 'res', 'sports', 'work_exp', 'leadership',
        'community_impact', 'ec_years', 'int_awards', 'nat_awards', 'state_awards', 'local_awards', 'other_awards'
    ])
    
    print("Original features:", df.columns.tolist())
    print("Number of original features:", len(df.columns))
    
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
    
    print("Features after engineering:", df.columns.tolist())
    print("Number of features after engineering:", len(df.columns))
    
    # Encoding categorical variables
    # Define all possible categories for each variable
    ethnicity_categories = [0, 1]  # Assuming 6 possible ethnicities
    gender_categories = [0, 1, 2]  # Assuming binary gender
    type_school_categories = [0, 1, 2, 3, 4]  # Assuming 3 types of schools
    location_categories = [0, 1, 2]  # Assuming 4 possible locations

    # One-hot encode with all possible categories
    df = pd.get_dummies(df, columns=['ethnicity', 'gender', 'type_school', 'location'])
    
    # Ensure all categories are present
    for cat in ethnicity_categories:
        if f'ethnicity_{cat}' not in df.columns:
            df[f'ethnicity_{cat}'] = 0
    for cat in gender_categories:
        if f'gender_{cat}' not in df.columns:
            df[f'gender_{cat}'] = 0
    for cat in type_school_categories:
        if f'type_school_{cat}' not in df.columns:
            df[f'type_school_{cat}'] = 0
    for cat in location_categories:
        if f'location_{cat}' not in df.columns:
            df[f'location_{cat}'] = 0
    
    print("Features after one-hot encoding:", df.columns.tolist())
    print("Number of features after one-hot encoding:", len(df.columns))
    
    return df

def main():
    # Load the saved model and scaler
    model = joblib.load('best_model_xgb.joblib')
    scaler = joblib.load('scaler.joblib')
    
    print("Number of features expected by scaler:", scaler.n_features_in_)
    
    # Load and preprocess the input data
    X, actual_accept_rate = load_data('train/inference/gpt-4o_output.json')
    X = feature_engineering(X)
    
    print("\nFinal number of features:", X.shape[1])
    print("Final feature names:", X.columns.tolist())
    
    # Scale the features
    try:
        X_scaled = scaler.transform(X)
    except ValueError as e:
        print(f"Error during scaling: {str(e)}")
        print(f"Number of features in scaler: {scaler.n_features_in_}")
        print(f"Number of features in current data: {X.shape[1]}")
        return
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    
    # Print results
    print(f"\nPredicted accept rate: {prediction:.2f}")
    
    if actual_accept_rate is not None:
        print(f"Actual accept rate: {actual_accept_rate:.2f}")
        print(f"Difference: {abs(prediction - actual_accept_rate):.2f}")
    else:
        print("Actual accept rate not provided in the input data.")

if __name__ == "__main__":
    main()