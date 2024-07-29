import json
from dotenv import load_dotenv
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import sys
import os
import tensorflow as tf
import math
import anthropic

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import load_data, preprocess_data

load_dotenv()
client = anthropic.Anthropic()

with open("/home/ubuntu/nochances/categorization/prompt.txt", "r") as f:
    prompt = f.read()

JSON_SCHEMAS = {
    "basic_info": {
        "ethnicity": "Integer: 0 = Underrepresented Minority in College (Black, Hispanic, Native American, Pacific Islander), 1 = Not Underrepresented Minority in College (White, Asian, Other). If multiple ethnicities are listed, use the one that would be considered underrepresented if applicable. Example: 1 for East Asian",
        "gender": "Integer: 0 = Male, 1 = Female, 2 = Other/Non-binary. If gender is not stated, do not assume based on other information, use 2 in this case. Example: 1 for Female",
        "income_bracket": "Integer: Family annual income pre-tax. 0 = $0-30k, 1 = $30k-60k, 2 = $60k-100k, 3 = $100k-200k, 4 = $200k+. If a range is given, use the middle value. If 'upper middle class' is mentioned without specifics, use 3. If 'lower income' or similar is mentioned without specifics, use 1. If not specified at all, use your expert knowledge to judge their activties and the costs associated to best guess their income bracket. Example: 0 for '<30K'",
        "type_school": "Integer: Most desired dream school type. 0 = STEM (e.g., MIT, Caltech), 1 = Liberal Arts (e.g., Williams, Amherst), 2 = Art/Design (e.g., RISD, Parsons), 3 = Music Conservatory (e.g., Juilliard, Berklee), 4 = Other specialized (e.g., military academies). For universities known for multiple areas, choose based on the applicant's intended major. Example: 0 for Harvard (considered STEM-strong) if applying for Biochemistry",
        "app_round": "Integer: For most desired dream school. 0 = Early Decision/Action, 1 = Regular Decision. If not specified, assume 1. Example: 1 for Regular Decision (if not specified)",
        "gpa": "Integer: Unweighted 4.0 scale. 0 = Below 2.5, 1 = 2.5-2.99, 2 = 3.0-3.49, 3 = 3.5-3.79, 4 = 3.8-4.0. If only weighted GPA is given, estimate unweighted by subtracting 0.5 (minimum 0), this is still above 4.0, just use 4. If no GPA is given but class rank is top 10%, use 4. Example: 4 for 4.0 UW",
        "ap_ib_courses": "Integer: Total number of AP and IB courses taken (exclude honors). If only total number of APs passed is given, use that number. Example: 10 for 10 APs",
        "ap_ib_scores": "Integer: 0 = No scores, 1 = Avg below 3(AP)/4(IB), 2 = Avg 3-3.9(AP)/4-4.9(IB), 3 = Avg 4-4.9(AP)/5-5.9(IB), 4 = Avg 5(AP)/6-7(IB). Calculate the average if multiple scores are given. If only some scores are mentioned, assume the others are average (3 for AP, 4 for IB). Example: 3 for mostly 4s and 5s on AP exams",
        "test_score": "Integer: 0 = No score/below 1000 SAT/20 ACT, 1 = 1000-1190 SAT/20-23 ACT, 2 = 1200-1390 SAT/24-29 ACT, 3 = 1400-1490 SAT/30-33 ACT, 4 = 1500+ SAT/34+ ACT. If both SAT and ACT are given, use the higher equivalent. If superscored, use that. Example: 4 for 1570 SAT or 35 ACT",
        "location": "Integer: 0 = Rural (<10k population), 1 = Suburban (10k-100k), 2 = Urban (>100k). If not explicitly stated, infer from context (e.g., 'small town' = 0, 'suburb of Chicago' = 1). Example: 1 for Suburban",
        "state_status": "Integer: For most desired dream school. 0 = In Same State, 1 = Not In Same State. If not explicitly stated, infer from context (e.g., if applying to many out-of-state schools, assume 1). Example: 1 for out-of-state (if not specified)",
        "legacy": "Integer: For most desired dream school. 0 = Legacy, 1 = Non-Legacy. Legacy includes parents, grandparents, or siblings who attended. If not mentioned, assume 1. Example: 1 for Non-Legacy (if not specified)",
        "intended_major": """Integer 1-10: Competitiveness of major at most desired dream school. Consider both the competitiveness of the major in general and at the specific school. Examples:
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
        "major_alignment": """Integer: 1-5 scale measuring how well the applicant's profile aligns with their intended major. Consider:
            - Relevance of coursework (e.g., advanced math/science for STEM majors)
            - Related extracurricular activities
            - Relevant awards or achievements
            - Research or projects in the field
            - Work or internship experience in the area
            Guidelines and Examples:
            1 = Minimal alignment
            Example: A student applying for Computer Science who has only taken standard high school math courses, has no coding experience, and whose extracurriculars are unrelated to tech (e.g., school band and community service at an animal shelter).

            2 = Basic alignment
            Example: A student applying for Biology who has taken AP Biology and Chemistry, participates in the school's science club, but has no research experience or biology-related projects. Their main extracurricular is being captain of the soccer team.

            3 = Moderate alignment
            Example: A student applying for English Literature who has taken AP English and several other literature courses, is an editor for the school newspaper, has won a local writing contest, and volunteers at the city library. However, they haven't pursued any independent writing projects or attended specialized writing programs.

            4 = Strong alignment
            Example: A student applying for Mechanical Engineering who has taken AP Physics, AP Calculus BC, and an intro to engineering course at a local community college. They're the team lead in their school's robotics club, have participated in multiple engineering competitions with good results, and completed a summer internship at a local manufacturing company.

            5 = Exceptional alignment
            Example: A student applying for Astrophysics who has exhausted their high school's math and physics courses and is taking multivariable calculus online. They've conducted research at a university lab, co-authored a paper in a youth science journal, won awards in national and international physics olympiads, and founded a popular astronomy club at their school that does public outreach events.

            Choose the level that best matches the applicant's alignment with their intended major based on these guidelines and examples.""",
        "first_gen": "Integer: 0 = Not first-generation college student, 1 = First-generation college student. First-gen means neither parent has a 4-year degree. If not explicitly stated, assume 0. Example: 0 (if not mentioned)",
        "languages": "Integer: Number of languages proficient in (including native). Count only if mentioned. Proficiency should be at conversational level or above. If not mentioned, use 1. Example: 1 (if not mentioned)",
        "special_talents": "Integer: 0 = None, 1 = School/local (e.g., lead in school play), 2 = Regional (e.g., first chair in regional orchestra), 3 = National (e.g., national chess champion), 4 = International (e.g., junior Olympian). Use highest level achieved. If multiple talents, use highest. Example: 2 for All-State Orchestra",
        "hooks": """Integer: Number of significant factors that may provide a notable advantage in the college admissions process. Guidelines:
            1) Include:
               - Recruited athlete (official recruitment by college coaches, not just participation in high school sports)
               - Child of faculty/staff at the most desired dream school
               - Significant hardship or adversity overcome (e.g., refugee status, homelessness, major health challenges)
               - Exceptional talent or achievement (e.g., published author, Olympic athlete, Carnegie Hall performer)
            2) Do not include:
               - Common experiences or challenges (e.g., divorce of parents, typical part-time job)
               - General diversity factors that don't significantly impact admissions (e.g., being left-handed, from a specific state)
               - Factors already accounted for in other categories (e.g., academic achievements, standard extracurricular activities, legacy)
            3) Counting:
               - Count each distinct hook as 1
               - Multiple hooks can be counted (e.g., Recruited athlete and significant hardship would count as 2)
               - For exceptional talents/achievements, only count if truly extraordinary and likely to significantly impact admissions
            4) If no hooks are explicitly mentioned or evident from the application, use 0
            Examples: 
               - 1 for recruited athlete
               - 2 for being both a child of faculty at most desired school and a recruited athlete
               - 3 for exceptional talent/achievement, child of faculty at most desired school, and overcoming a major health challenge""",
    },
    "ecs": {
        "nat_int": """Integer: Number of National or International Activities in notable positions. 'Notable' means:
            1) Leadership role in a national/international organization
            2) Significant achievement at a national/international level
            3) Sustained participation (>1 year) in a highly selective national/international program
        Examples:
            - Officer in a national student organization
            - Participant in a highly selective international summer program (e.g., RSI, TASP)
            - Member of a national sports team or youth orchestra
            - Intern at a major national/international company or research lab
        Do not include:
            - One-time participation in a national competition without significant achievement
            - Online courses or programs without rigorous selection processes
            - Local chapters of national organizations (count these in 'local')
        Count each distinct activity separately.""",
        "reg": """Integer: Number of State or Regional Activities in notable positions. 'Notable' means:
            1) Leadership role in a state/regional organization
            2) Significant achievement at a state/regional level
            3) Sustained participation (>1 year) in a selective state/regional program
        Examples:
            - All-State Orchestra or Choir member
            - Captain of a regional sports team
            - Leader in state-wide student government or youth organization
            - Participant in a selective state-wide academic program
        Do not include:
            - One-time participation in a state competition without placing
            - Activities already counted in 'nat-int'
        Count each distinct activity separately.""",
        "local": """Integer: Number of Local or School Activities/Clubs in notable positions. 'Notable' means:
            1) Leadership role in a school club or local organization
            2) Significant achievement or recognition at the school/local level
            3) Sustained participation (>1 year) with increasing responsibility
        Examples:
            - President, Vice President, or founder of a school club
            - Captain of a school sports team
            - Editor of school newspaper or literary magazine
            - Lead role in school theater productions
        Do not include:
            - General membership in clubs without leadership roles or significant contributions
            - One-time participation in local events without notable achievement
            - Activities already counted in 'nat-int' or 'reg'
        Count each distinct activity separately.""",
        "volunteering": "Integer: Number of distinct Volunteering Activities. Count each organization or cause separately. Example: 2 for volunteering at a dog shelter and a local library",
        "ent": "Integer: Number of Entrepreneurial ventures (founded/co-founded). Include both for-profit and non-profit initiatives. Example: 0 (if not mentioned)",
        "intern": "Integer: Number of internships. Include both paid and unpaid. Count each distinct internship, even if at the same company. Example: 0 (if not mentioned)",
        "add": """Integer: Number of significant academic classes or programs taken outside the standard high school curriculum. Guidelines:
            1) Include:
               - College-level courses taken at a university or community college
               - Accredited online college courses (e.g., through edX, Coursera if verified/certificate track)
               - Structured summer academic programs at recognized institutions
               - Formal, multi-week academic programs (e.g., intensive language programs, coding bootcamps)
            2) Do not include:
               - AP or IB classes (these are counted elsewhere)
               - Regular high school classes, even if advanced
               - Brief workshops or one-day seminars
               - Informal, self-guided online learning
            3) Counting:
               - Count each semester-long course or equivalent as 1
               - For year-long courses, count as 2
               - For intensive summer programs, count each distinct program as 1
               - If only a total number of college credits is given, divide by 3 and round down to estimate number of courses
            4) If not explicitly mentioned in the application, use 0
            Example: 3 for taking two semester-long college math courses and completing a 6-week summer coding bootcamp""",
        "res": """Integer: Number of significant research projects or papers that are completed or substantively in-progress. Guidelines:
            1) Include:
               - Independent research projects with clear objectives and methodology
               - Collaborative research as part of a team (e.g., in a university lab or group)
               - Mentored research programs (e.g., science fair projects with professional mentorship)
               - Published or unpublished papers resulting from research
               - Ongoing research projects that have produced preliminary results or a defined research plan
            2) Do not include:
               - Brief (less than a month) research experiences or shadowing
               - School assignments or projects that are part of regular coursework
               - Vague mentions of 'interest in research' without specific projects
            3) Counting:
               - Count each distinct research project or paper as 1
               - For ongoing multi-year projects, still count as 1 unless it has resulted in multiple distinct papers or presentations
               - If a single project results in both a paper and a presentation, count it as 1
               - Summer research programs count as 1, unless they resulted in multiple distinct projects
            4) If research experience is not explicitly mentioned in the application, use 0
            Examples: 
               - 1 for a year-long neuroscience research project in a university lab
               - 2 for an independent ecology study resulting in a paper, plus a summer research program in chemistry
               - 3 for two published papers from different projects, plus an ongoing research project with preliminary results""",
        "sports": "Integer: Number of sports teams (e.g., JV, Varsity). Count each sport separately, even if played multiple years. Example: 0 (if not mentioned)",
        "work_exp": "Integer: Number of part-time jobs or significant family responsibilities. Count each distinct job or responsibility. Example: 1 for working as a private violin tutor",
        "leadership": "Integer: Total number of leadership positions across all activities. Include positions like president, vice president, captain, editor, etc. Example: 3 for being president of 3 clubs",
        "community_impact": """Integer: Judge based on scope, duration, and measurable impact of all activities combined. Examples:
            0 = None (no community service or impact mentioned)
            1 = Minimal (e.g., occasional volunteering at local food bank, participating in school cleanup day)
            2 = Moderate (e.g., regular volunteering at hospital, leading a school recycling program)
            3 = Significant (e.g., founding a club that provides tutoring to underprivileged students, organizing a large-scale community fundraiser)
            4 = Exceptional with measurable impact (e.g., creating a city-wide program that significantly increased youth literacy rates, founding a non-profit that provided measurable aid to a large number of people)""",
        "ec_years": "Integer: Average years of involvement across significant extracurriculars. Sum total years of all activities and divide by number of activities. Round to nearest whole number. Example: 4 if involved in most activities for all 4 years of high school",
    },
    "awards": {
        "int": "Integer: Number of top-tier International Awards. Only include awards for placing 1st-10th (or equivalent prestigious recognition) in global competitions with participants from multiple countries. Examples: International Math Olympiad medalist, International Science and Engineering Fair (ISEF) Grand Award winner. Do not include qualifications or lower-level achievements in international competitions (e.g., USACO Bronze/Silver/Gold, International Chemistry Olympiad participant without placing).",
        "nat": "Integer: Number of National Awards (1st-10th place or equivalent national recognition). Include nationwide competitions, national merit, etc. Examples: National Merit Finalist, USAMO qualifier, USACO Platinum, Presidential Scholar. Do not include awards already counted in 'int'.",
        "state": "Integer: Number of State/Regional Awards (1st-10th place or equivalent). Include state competitions, regional recognitions, etc. Examples: All-State Orchestra, State Science Fair winner, Regional Math Olympiad top performer. Do not include awards already counted in 'int' or 'nat'.",
        "local": "Integer: Number of significant Local/School Awards (typically 1st-3rd place). Include school competitions, local recognitions, etc. Examples: Valedictorian, School Science Fair winner, District Math Competition top 3. Do not include awards already counted in higher categories.",
        "other": "Integer: Number of Honorable Mentions, Participation Awards, or minor recognitions at any level. Include recognitions that don't fit in above categories or place below 3rd in local competitions. Examples: AP Scholar, Honor Roll, Participation in USACO Bronze/Silver without advancing, School 'Student of the Month'.",
    },
}

json_string = json.dumps(JSON_SCHEMAS, indent=2)


def process_post_with_gpt(post):
    try:
        completion = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1000,
            temperature=0,    
            system = prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f'Analyze the following Reddit post from r/collegeresults and extract the relevant information according to the specified JSON schema. If the post lacks sufficient information to provide an accurate, consistent output, contains clearly false information, or is a joke, output only {{"skip": true}}. If some information is not explicitly stated, make your best reasonable inference based on context. However, if too much critical information is missing, output only {{"skip": true}}. Do not include any dialogue or explanations, only output valid JSON. You must output valid JSON in this format: {json_string}\nHere is the Reddit Post to analyze:\n{post}',
                        }
                    ]
                },
            ],
        )
        
        # Print the raw content for debugging
        print("Raw GPT response:", completion.content)
        
        # Extract text from TextBlock if necessary
        if isinstance(completion.content, list) and len(completion.content) > 0 and hasattr(completion.content[0], 'text'):
            content = completion.content[0].text
        else:
            content = completion.content
        
        # Check if content is already a dictionary
        if isinstance(content, dict):
            return content
        
        # If it's a string, try to parse it as JSON
        try:
            parsed_content = json.loads(content)
            return parsed_content
        except json.JSONDecodeError as json_error:
            print(f"Error parsing JSON: {json_error}")
            return None

    except Exception as e:
        print(f"Error processing post with GPT: {e}")
        return None


def get_school_acceptance_rate_category(school_name, intended_major):
    try:
        completion = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=100,
            temperature=0,
            system="You are an AI assistant with extensive knowledge of university acceptance rates. Your task is to categorize universities based on their most recent publicly available overall acceptance rate. Respond ONLY with the appropriate category number (0, 1, 2, or 3) and nothing else.",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"""
Categorize {school_name}'s acceptance rate using the following scale:
0 = <5% (e.g., Harvard, Stanford, MIT)
1 = 5-15% (e.g., Northwestern, Cornell)
2 = 15-40% (e.g., UC Davis, Boston University)
3 = >40% (e.g., ASU, Rollins University) or Open Admission
Use the most recent publicly available overall acceptance rate for initial categorization. Then, adjust the category if the intended major ({intended_major}) is known to be significantly more competitive at {school_name} Only change school category if the change would result in a lower integer.
Examples of major-specific adjustments:

Computer Science at Carnegie Mellon: Category 0
EECS at UC Berkeley: Category 0
Engineering at Georgia Tech: Lower category than overall admission

Consider factors like:

Department-specific acceptance rates (if available)
Reputation of the program within the field
Historical data on major competitiveness

Return only the integer category (0, 1, 2, or 3) that best represents the difficulty of admission for {intended_major} at {school_name}.
"""
                }]
                },
            ],
        )
        
        # Extract text from TextBlock if necessary
        if isinstance(completion.content, list) and len(completion.content) > 0 and hasattr(completion.content[0], 'text'):
            content = completion.content[0].text
        else:
            content = completion.content
        
        # Print the raw content for debugging
        print(f"Raw school category response: {content}")
        
        # Strip whitespace and convert to integer
        return int(content.strip())
    except Exception as e:
        print(f"Error getting school acceptance rate category: {e}")
        return None

def load_data_from_json(json_data):
    if json_data is None or "skip" in json_data:
        return None

    features = []
    for category, values in json_data.items():
        for key, value in values.items():
            if isinstance(value, str) and value.isdigit():
                value = int(value)
            features.append(value)

    return np.array([features])


def calculate_acceptance_probability(ensemble_prediction, school_category):
    # Adjust the prediction based on the school category
    adjusted_prediction = ensemble_prediction - school_category

    # Define the logistic function parameters
    k = 1
    x0 = 0.5

    # Calculate the base probability using a modified logistic function
    base_probability = 1 / (1 + math.exp(k * (adjusted_prediction - x0)))

    # Apply additional adjustments
    if adjusted_prediction <= 0:
        # Boost probabilities for predictions at or below the school category
        boost_factor = 1 + abs(adjusted_prediction) * 0.1
        probability = base_probability * boost_factor
    else:
        # Reduce probabilities for predictions above the school category
        reduction_factor = 1 - min(adjusted_prediction * 0.1, 0.5)
        probability = base_probability * reduction_factor

    # Ensure the probability stays within [0.01, 0.99] range
    probability = max(0.01, min(0.99, probability))

    return probability

def get_color_for_probability(probability):
    if probability < 0.3:
        return "#D34C45"  # Red
    elif probability < 0.5:
        return "#E68128"  # Orange
    elif probability < 0.7:
        return "#D6AD3D"  # Yellow
    elif probability < 0.9:
        return "#6CB054"  # Green
    else:
        return "#4E9AC1"  # Blue
    
def predict_acceptance(post, school_name, major):
    # Process the post with GPT
    gpt_output = process_post_with_gpt(post)

    if gpt_output is None or "skip" in gpt_output:
        return "Unable to process the post or insufficient information provided."

    # Load and preprocess the input data
    X = load_data_from_json(gpt_output)
    if X is None:
        return "Unable to extract features from the processed data."

    # Load the saved models and scaler
    xgb_model = joblib.load("/home/ubuntu/nochances/models/best_model_xgb.joblib")
    nn_model = tf.keras.models.load_model("/home/ubuntu/nochances/models/best_model_nn.keras")
    scaler = joblib.load("/home/ubuntu/nochances/models/scaler.joblib")

    # Preprocess the input data
    X_preprocessed = preprocess_data(X, is_training=False, scaler=scaler)

    # Make predictions
    xgb_prediction = xgb_model.predict(X_preprocessed)[0]
    nn_prediction = nn_model.predict(X_preprocessed).flatten()[0]

    # Get the school's acceptance rate category
    school_category = get_school_acceptance_rate_category(school_name, major)

    if school_category is None:
        return "Unable to determine the school's acceptance rate category."

    # Calculate the probability of acceptance
    probability = calculate_acceptance_probability(xgb_prediction, school_category)
    # Convert non-serializable types
    result = {
        "school_category": int(school_category),
        "acceptance_probability": float(probability),
        "xgb_prediction": float(xgb_prediction),
        "color": get_color_for_probability(probability)
    }

    return result