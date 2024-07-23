#inference.py
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
        "ethnicity": "Integer: 0 = Underrepresented Minority in College (Black, Hispanic, Native American, Pacific Islander), 1 = Not Underrepresented Minority in College (White, Asian, Other). If multiple ethnicities are listed, use the one that would be considered underrepresented if applicable. Example: 1 for East Asian",
        "gender": "Integer: 0 = Male, 1 = Female, 2 = Other/Non-binary. If gender is not stated, do not assume based on other information, use 2 in this case. Example: 1 for Female",
        "income_bracket": "Integer: Family annual income pre-tax. 0 = $0-30k, 1 = $30k-60k, 2 = $60k-100k, 3 = $100k-200k, 4 = $200k+. If a range is given, use the middle value. If 'upper middle class' is mentioned without specifics, use 3. If 'lower income' or similar is mentioned without specifics, use 1. Example: 0 for '<30K'",
        "type_school": "Integer: Most selective accepted school type. 0 = STEM (e.g., MIT, Caltech), 1 = Liberal Arts (e.g., Williams, Amherst), 2 = Art/Design (e.g., RISD, Parsons), 3 = Music Conservatory (e.g., Juilliard, Berklee), 4 = Other specialized (e.g., military academies). For universities known for multiple areas, choose based on the applicant's intended major. Example: 0 for Harvard (considered STEM-strong) if applying for Biochemistry",
        "app_round": "Integer: For most selective accepted school. 0 = Early Decision/Action, 1 = Regular Decision. If not specified, assume 1. Example: 1 for Regular Decision (if not specified)",
        "gpa": "Integer: Unweighted 4.0 scale. 0 = Below 2.5, 1 = 2.5-2.99, 2 = 3.0-3.49, 3 = 3.5-3.79, 4 = 3.8-4.0. If only weighted GPA is given, estimate unweighted by subtracting 0.5 (minimum 0), this is still above 4.0, just use 4. If no GPA is given but class rank is top 10%, use 4. Example: 4 for 4.0 UW",
        "ap-ib-courses": "Integer: Total number of AP and IB courses taken (exclude honors). If only total number of APs passed is given, use that number. Example: 10 for 10 APs",
        "ap-ib-scores": "Integer: 0 = No scores, 1 = Avg below 3(AP)/4(IB), 2 = Avg 3-3.9(AP)/4-4.9(IB), 3 = Avg 4-4.9(AP)/5-5.9(IB), 4 = Avg 5(AP)/6-7(IB). Calculate the average if multiple scores are given. If only some scores are mentioned, assume the others are average (3 for AP, 4 for IB). Example: 3 for mostly 4s and 5s on AP exams",
        "test-score": "Integer: 0 = No score/below 1000 SAT/20 ACT, 1 = 1000-1190 SAT/20-23 ACT, 2 = 1200-1390 SAT/24-29 ACT, 3 = 1400-1490 SAT/30-33 ACT, 4 = 1500+ SAT/34+ ACT. If both SAT and ACT are given, use the higher equivalent. If superscored, use that. Example: 4 for 1570 SAT or 35 ACT",
        "location": "Integer: 0 = Rural (<10k population), 1 = Suburban (10k-100k), 2 = Urban (>100k). If not explicitly stated, infer from context (e.g., 'small town' = 0, 'suburb of Chicago' = 1). Example: 1 for Suburban",
        "state-status": "Integer: For most selective accepted school. 0 = In Same State, 1 = Not In Same State. If not explicitly stated, infer from context (e.g., if applying to many out-of-state schools, assume 1). Example: 1 for out-of-state (if not specified)",
        "legacy": "Integer: For most selective accepted school. 0 = Legacy, 1 = Non-Legacy. Legacy includes parents, grandparents, or siblings who attended. If not mentioned, assume 1. Example: 1 for Non-Legacy (if not specified)",
        "intended_major": """Integer 1-10: Competitiveness of major at most selective accepted school. Consider both the competitiveness of the major in general and at the specific school. Examples:
            1 = Least competitive (e.g., Liberal Studies at NYU)
            2 = Very low competitiveness (e.g., Anthropology at UC Berkeley)
            3 = Low competitiveness (e.g., English at UCLA)
            4 = Below average competitiveness (e.g., History at UMich)
            5 = Average competitiveness (e.g., Psychology at Cornell)
            6 = Above average competitiveness (e.g., Economics at Duke)
            7 = High competitiveness (e.g., Mechanical Engineering at Georgia Tech)
            8 = Very high competitiveness (e.g., Computer Science at Carnegie Mellon)
            9 = Extremely high competitiveness (e.g., Bioengineering at MIT)
            10 = Most competitive (e.g., Computer Science at Stanford)
        Use the closest match based on the school's reputation for the specific major.""",        
        "first_gen": "Integer: 0 = Not first-generation college student, 1 = First-generation college student. First-gen means neither parent has a 4-year degree. If not explicitly stated, assume 0. Example: 0 (if not mentioned)",
        "languages": "Integer: Number of languages proficient in (including native). Count only if mentioned. Proficiency should be at conversational level or above. If not mentioned, use 1. Example: 1 (if not mentioned)",
        "special_talents": "Integer: 0 = None, 1 = School/local (e.g., lead in school play), 2 = Regional (e.g., first chair in regional orchestra), 3 = National (e.g., national chess champion), 4 = International (e.g., junior Olympian). Use highest level achieved. If multiple talents, use highest. Example: 2 for All-State Orchestra",
        "hooks": "Integer: Number of significant 'hooks' (e.g., recruited athlete, development case, child of faculty, low-income, first-gen, URM). Count each distinct hook. Example: 1 for low-income status",
        "accept_rate": "Integer: Selectivity of most selective accepted school. 0 = <5% (e.g., Harvard, Stanford, MIT), 1 = 5-15% (e.g., Northwestern, Cornell), 2 = 15-40% (e.g., UC Davis, Boston University), 3 = >40% (e.g., ASU) or Open Admission. Use most recent publicly available data. Example: 0 for Harvard"
    },
    "ecs": {
        "nat-int": "Integer: Number of National/International Activities in notable position. 'Notable' means leadership role, significant achievement, or sustained participation (>1 year). Count each distinct activity. Example: 1 for participation in a national-level orchestra",
        "reg": "Integer: Number of State/Regional Activities in notable position. 'Notable' as defined above. Count each distinct activity. Example: 1 for All-State Symphony Orchestra",
        "local": "Integer: Number of Local/School Activities/Clubs in notable position. 'Notable' as defined above. Count each distinct activity. Example: 3 for being president of 3 community service clubs",
        "volunteering": "Integer: Number of distinct Volunteering Activities. Count each organization or cause separately, even if done through school clubs. Example: 1 for 100+ hours of volunteering",
        "ent": "Integer: Number of Entrepreneurial ventures (founded/co-founded). Include both for-profit and non-profit initiatives. Example: 0 (if not mentioned)",
        "intern": "Integer: Number of internships. Include both paid and unpaid. Count each distinct internship, even if at the same company. Example: 0 (if not mentioned)",
        "add": "Integer: Number of classes taken outside high school curriculum. Include college courses, online certifications, etc. Count each distinct course. Example: 0 (if not mentioned)",
        "res": "Integer: Number of research projects or papers completed. Include both independent and mentored research. Count each distinct project. Example: 1 for Neuroscience Student Researcher",
        "sports": "Integer: Number of sports teams (e.g., JV, Varsity). Count each sport separately, even if played multiple years. Example: 0 (if not mentioned)",
        "work_exp": "Integer: Number of part-time jobs or significant family responsibilities. Count each distinct job or responsibility. Example: 1 for working as a private violin tutor",
        "leadership": "Integer: Total number of leadership positions across all activities. Include positions like president, vice president, captain, editor, etc. Example: 3 for being president of 3 clubs",
        "community_impact": """Integer: Judge based on scope, duration, and measurable impact of activities. Examples:
            0 = None (no community service or impact mentioned)
            1 = Minimal (e.g., occasional volunteering at local food bank, participating in school cleanup day)
            2 = Moderate (e.g., regular volunteering at hospital, leading a school recycling program)
            3 = Significant (e.g., founding a club that provides tutoring to underprivileged students, organizing a large-scale community fundraiser)
            4 = Exceptional with measurable impact (e.g., creating a city-wide program that significantly increased youth literacy rates, founding a non-profit that provided measurable aid to a large number of people)""",
        "ec_years": "Integer: Average years of involvement across significant extracurriculars. Sum total years of all activities and divide by number of activities. Round to nearest whole number. Example: 4 if involved in most activities for all 4 years of high school"
    },
    "awards": {
        "int": "Integer: Number of International Awards (1st-10th place or equivalent). Include olympiads, global competitions, etc. Count each distinct award. If an award qualifies for multiple categories, only count it here. Example: 1 for International Math Olympiad Silver Medal",
        "nat": "Integer: Number of National Awards (1st-10th place or equivalent). Include nationwide competitions, national merit, etc. Count each distinct award. Only count here if not already counted in 'int'. Example: 1 for National Merit Finalist",
        "state": "Integer: Number of State/Regional Awards (1st-10th place or equivalent). Include state competitions, regional recognitions, etc. Count each distinct award. Only count here if not already counted in 'int' or 'nat'. Example: 1 for All-State Orchestra (4 years counts as 1 award)",
        "local": "Integer: Number of Local/School Awards (1st-3rd place or equivalent). Include school competitions, local recognitions, etc. Count each distinct award. Only count here if not already counted in higher categories. Example: 1 for valedictorian",
        "other": "Integer: Number of Honorable Mentions or Participation Awards (any level, no duplicates). Include recognitions that don't fit in above categories or place below 3rd in local competitions. Count each distinct recognition. Example: 1 for Honorable Mention in a national essay contest"
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
                    {"role": "user", "content": f"Analyze the following Reddit post from r/collegeresults and extract the relevant information according to the specified JSON schema. If the post lacks sufficient information to provide an accurate, consistent output, contains clearly false information, or is a joke, output only {{\"skip\": true}}. If some information is not explicitly stated, make your best reasonable inference based on context. However, if too much critical information is missing, output only {{\"skip\": true}}. Do not include any dialogue or explanations, only output valid JSON. You must output valid JSON in this format: {json_string}\nHere is the Reddit Post to analyze: {flair}\n{post}"}
                ],
                temperature=0.0,
                seed=42,
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
    ethnicity_categories = [0, 1]
    gender_categories = [0, 1, 2]
    type_school_categories = [0, 1, 2, 3, 4]
    location_categories = [0, 1, 2]

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