from openai import OpenAI
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

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing import load_data, preprocess_data

load_dotenv()
client = OpenAI()

with open("categorization/prompt.txt", "r") as f:
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
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f'Analyze the following Reddit post from r/collegeresults and extract the relevant information according to the specified JSON schema. If the post lacks sufficient information to provide an accurate, consistent output, contains clearly false information, or is a joke, output only {{"skip": true}}. If some information is not explicitly stated, make your best reasonable inference based on context. However, if too much critical information is missing, output only {{"skip": true}}. Do not include any dialogue or explanations, only output valid JSON. You must output valid JSON in this format: {json_string}\nHere is the Reddit Post to analyze:\n{post}',
                },
            ],
            temperature=0.0,
            seed=42,
            response_format={"type": "json_object"},
        )
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(f"Error processing post with GPT: {e}")
        return None


def get_school_acceptance_rate_category(school_name):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in college admissions with up-to-date knowledge of acceptance rates for various universities.",
                },
                {
                    "role": "user",
                    "content": f"Given the following categories for college acceptance rates:\n\n0 = <5% (e.g., Harvard, Stanford, MIT)\n1 = 5-15% (e.g., Northwestern, Cornell)\n2 = 15-40% (e.g., UC Davis, Boston University)\n3 = >40% (e.g., ASU, Rollins University) or Open Admission\n\nPlease categorize {school_name} based on its most recent publicly available acceptance rate. Return only the integer category (0, 1, 2, or 3).",
                },
            ],
            temperature=0.0,
            seed=42,
        )
        return int(completion.choices[0].message.content.strip())
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
    k = 3
    x0 = 0.5

    # Calculate the base probability using a modified logistic function
    base_probability = 1 / (1 + math.exp(k * (adjusted_prediction - x0)))

    # Apply additional adjustments
    if adjusted_prediction <= 0:
        # Boost probabilities for predictions at or below the school category
        boost_factor = 1 + abs(adjusted_prediction) * 0.2
        probability = base_probability * boost_factor
    else:
        # Reduce probabilities for predictions above the school category
        reduction_factor = 1 - min(adjusted_prediction * 0.1, 0.5)
        probability = base_probability * reduction_factor

    # Ensure the probability stays within [0.01, 0.99] range
    probability = max(0.01, min(0.99, probability))

    return probability


def predict_acceptance(post, school_name):
    # Process the post with GPT
    gpt_output = process_post_with_gpt(post)

    if gpt_output is None or "skip" in gpt_output:
        return "Unable to process the post or insufficient information provided."

    # Load and preprocess the input data
    X = load_data_from_json(gpt_output)
    if X is None:
        return "Unable to extract features from the processed data."

    # Load the saved models and scaler
    xgb_model = joblib.load("models/best_model_xgb.joblib")
    nn_model = tf.keras.models.load_model("models/best_model_nn.keras")
    scaler = joblib.load("models/scaler.joblib")

    # Preprocess the input data
    X_preprocessed = preprocess_data(X, is_training=False, scaler=scaler)

    # Make predictions
    xgb_prediction = xgb_model.predict(X_preprocessed)[0]
    nn_prediction = nn_model.predict(X_preprocessed).flatten()[0]

    # Ensemble prediction (average of XGBoost and Neural Network)
    ensemble_prediction = (xgb_prediction + nn_prediction) / 2

    # Get the school's acceptance rate category
    school_category = get_school_acceptance_rate_category(school_name)

    if school_category is None:
        return "Unable to determine the school's acceptance rate category."

    # Calculate the probability of acceptance
    probability = calculate_acceptance_probability(ensemble_prediction, school_category)

    return {
        "ensemble_prediction": ensemble_prediction,
        "school_category": school_category,
        "acceptance_probability": probability,
        "nn_prediction": nn_prediction,
        "xgb_prediction": xgb_prediction,
    }


if __name__ == "__main__":
    example_post = """
3.8+|1500+/34+|STEM
Demographics

Gender: female

Race/Ethnicity: white

Residence: WA state (not Seattle area)

Income Bracket: ~60K, ~180K, or ~200K (depends on fin aid policy; custodial+noncustodial, custodial+step-parent, or all three).

Type of School: non-competitive public that sends 1-5 people per year to private T35s

Hooks (Recruited Athlete, URM, First-Gen, Geographic, Legacy, etc.): none (LGBT+, would be low income without recent step-parent, area is underrepresented even though state is overrepresented).

Intended Major(s): chemistry, applied with possible minors/double majors in econ, poli sci, history of science & technology, Spanish, etc. Major was always the same, minors were just what I was vibing with that day.

Academics

GPA (UW/W): 3.98 UW (~4.55 W, school does not weight GPAs or use a weighted GPA for anything)

Rank (or percentile): 61/456 (unweighted ranking system)

# of Honors/AP/IB/Dual Enrollment/etc.: 9 APs, 1 post-AP, 2 honors, 4 dual enrollment, 5 years of Spanish

Senior Year Course Load: AP Physics 1, AP U.S. Government, Principles of Biomedical Science, Multivariable Calculus, AP English Literature, Spanish V

Standardized Testing

List the highest scores earned and all scores that were reported.

SAT I: 1450 (740RW, 710M; did not report)

ACT: 34 (35E, 31M, 36R, 33S)

AP/IB: 5/4/4/4/4/3/3

Extracurriculars/Activities

List all extracurricular involvements, including leadership roles, time commitments, major achievements, etc.

Archery (competed nationally, rangemaster/assist coaches at range, founding member of JOAD club, a slew of 1st/2nd places at regional tournaments. 10 hours/week)

Speech & Debate (congressional debate + informative speaking, lettered varsity and “debate mom”, competed at state in Congress, nothing really high impact. 6 hours/week during tournament season).

Community service club vice president (organized a lot of events, this club has also existed for 5+ years, put around 5 hours/week into this).

Math team (local competitions, minor awards, tutoring, etc. 4 hours/week).

GSA club (officer, four year member, helped lead projects that got the club involved with the school and greater community. Led walkout after a bad incident of homophobia took place at our school. 2 hours/week).

Foreign language learning (mainly Spanish, German, and Polish, 7 hours/week outside of school throughout the year; this is pretty leisurely for me).

Chemistry club treasurer/founding member (club only lasted a year as the supervisor got too busy for it; we drew up actual lab proposals and gathered the supplies to complete them, lots of time spent researching. As treasurer, I organized several fundraisers, managed club funds, and approved/made all purchases).

Art stuffs (lots of drawing and painting, never entered any contests though I did submit a portfolio to every college that would take them. Varies significantly, averages to 2 hours/week).

Other volunteering activity (16 hours/week over the summer)

Random club that didn’t do much, technically a founding member (pretty much filler; 1 hour/week).

Awards/Honors

List all awards and honors submitted on your application.

Poetry contest winner (national)

Scholarship winner (regional)

Seal of Biliteracy (Spanish)

AP Scholar w/ Distinction

National Honor Society

Letters of Recommendation

English teacher, 9/10. We got along great and she adored my writing (used my essays most of the time when showing examples), read the letter and it felt pretty glowing.

Chemistry teacher, 8/10. Again, got along great with her and I was definitely an over-achiever in her class. Didn’t get to read it and she isn’t the strongest of writers, so 8/10 is a solid guess.

History teacher, 6/10. We got along great, he openly brags about me, however he used the same recommendation outline that he used for a student last year. Granted, that student got into Notre Dame, but wasn’t nearly as personalized.

Spanish teacher, 7.5/10. I’m one of her standouts and we know each other quite well, been to her house for a barbecue before. Read the letter and it wasn’t as glowing as my English teacher’s, but it highlighted a lot of achievements that I couldn’t put elsewhere in my application. Content was great, wording/tone had room to improve.

Interviews

Bryn Mawr, 2/10. My interviewer and I were not vibing, I was incredibly nervous, and after I consoled myself over how I just blew my chances of admission to a hard target-ish school that I really liked.

Mount Holyoke, 9/10. Aside from a little misunderstanding that I didn’t correct, we ~vibed~. Loved my interviewer, she made me want to attend more than I had wanted to before the interview!

Swarthmore, 7/10. Mostly vibed, but we didn’t have the same click I had with other interviewers. Good interview and conversation.

Princeton, 9/10. Interviewer was a super cool guy and we had an awesome conversation. Their interviews aren’t used in the admissions process, however, so I just appreciated it for what it was.

MIT, 0/10. Hated my interviewer who was a sexist mfer, came very close to leaving the call early. Made many backhanded comments about women at MIT, my major, my city, and even MIT in general. Very much a self-important dude bro, made me regret applying to MIT in general. If I’d been admitted, he gave me such an ick that I genuinely wouldn’t have attended.

Essays

Personal statement was probably the best essay I wrote throughout the cycle. Spent 20+ hours on it and revised a lot between November and January. It was about my speech impediment and debate, a story which my debate coach found very inspirational.

My essays for Georgia Tech were heavily targeting their mission statement in how I framed everything. After submitting, I felt incredibly confident about my admission. Spent maybe 5 hours total on this application.

Other supplements were largely done the day of the deadline and very rushed, though quality was still fairly high. I write best under a deadline, which unfortunately does jive overly well with being proactive about college applications.

Want to go to mit
    """

    school_name = "mit"
    prediction = predict_acceptance(example_post, school_name)
    print(f"Prediction for {school_name}:")
    print(f"NN prediction: {prediction['nn_prediction']:.2f}")
    print(f"XGB prediction: {prediction['xgb_prediction']:.2f}")
    print(f"Ensemble prediction: {prediction['ensemble_prediction']:.2f}")
    print(f"School category: {prediction['school_category']}")
    print(f"Probability of acceptance: {prediction['acceptance_probability']:.2%}")
