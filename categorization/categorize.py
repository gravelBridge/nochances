from openai import OpenAI
import json
from dotenv import load_dotenv
import time
import threading
from queue import Queue
import concurrent.futures

load_dotenv()
client = OpenAI()

with open('scraping/combined_collegeresults_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

post_ids = list(data.keys())

with open('categorization/prompt.txt', 'r') as f:
    prompt = f.read()

def parse_and_append(json_string, filename):
    data = json.loads(json_string)
    with open(filename, 'a') as file:
        json.dump(data, file)
        file.write('\n')


JSON_SCHEMAS = {
    "basic_info": {
        "ethnicity": "One integer number where 0: Underrepresented Minority (Black, Hispanic, Native American, Pacific Islander) or 1: Not Underrepresented Minority (White, Asian, Other)",
        "gender": "One integer number where 0: Male, 1: Female, or 2: Other/Non-binary",
        "income_bracket": "Family annual income pre-tax One integer number where 0: $0-30,000, 1: $30,001-60,000, 2: $60,001-100,000, 3: $100,001-200,000, 4: $200,001+ If not specified, make your best guess based on what they do",
        "type_school": "Out of all the schools they were accepted into, what type of school is the most selective accepted school? One integer number where 0: STEM-focused institution, 1: Liberal Arts institution, 2: Art/Design school, 3: Music Conservatory, 4: Other specialized institution",
        "app_round": "For the most selective school they got in to, which round did they apply in? One integer number where 0: Early Decision/Early Action, 1: Regular Decision",
        "gpa": "Unweighted, 4.0 scale, estimate in a 4.0 UW scale if not available in this format One integer number where 0: Below 2.5, 1: 2.5 to 2.99, 2: 3.0 to 3.49, 3: 3.5 to 3.79, 4: 3.8 to 4.0",
        "ap-ib-courses": "One integer number where that number is the total number of AP and IB courses taken, only AP and IB, do not include honors",
        "ap-ib-scores": "One integer number where 0: No AP/IB scores, 1: Average score below 3 (AP) or 4 (IB), 2: Average score 3-3.9 (AP) or 4-4.9 (IB), 3: Average score 4-4.9 (AP) or 5-5.9 (IB), 4: Average score 5 (AP) or 6-7 (IB)",
        "test-score": "One integer number where 0: No score or below 1000 SAT / 20 ACT, 1: 1000-1190 SAT / 20-23 ACT, 2: 1200-1390 SAT / 24-29 ACT, 3: 1400-1490 SAT / 30-33 ACT, 4: 1500+ SAT / 34+ ACT",
        "location": "One integer number where 0: Rural (population < 10,000), 1: Suburban (population 10,000 - 100,000), 2: Urban (population > 100,000)",
        "state-status": "For the most selective school they got in to, do they live in the same state as that school? One integer number where 0: In Same State 1: Not In Same State",
        "legacy": "For the most selective school they got in to, are they a legacy student at that school? One integer number where 0: Legacy, 1: Non-Legacy",
        "intended_major": "How competitive is their intended major at the most selective school they got in to on a scale from 1-10 (including 1 and 10)? Where 1 is not at all competitive, like gender studies at MIT, and 10 being CS at MIT. One integer value.",
        "first_gen": "Is the student a first-generation college student? One integer number where 0: No, 1: Yes If not specified, say 0",
        "languages": "Number of languages the student is proficient in (including native language). One integer value. If not specified, say 1",
        "special_talents": "One integer number representing the level of exceptional talent in a specific area (e.g., music, art, specific academic field) where 0: No exceptional talent, 1: School/local level talent, 2: Regional level talent, 3: National level talent, 4: International level talent If not specified, say 0",
        "hooks": "One integer number representing the number of significant 'hooks' the applicant has (e.g., recruited athlete, development case, child of faculty)",
        "accept_rate": "For the most selective school the student was accepted in to, how selective is that most selective school they were *accepted* into? One integer number where 0: Highly Selective (acceptance rate <5% (e.g. MIT)), 1: Very Selective (acceptance rate 5-15% (e.g. UCLA)), 2: Selective (acceptance rate 15-40% (e.g. UC Davis)), 3: Minimally Selective (acceptance rate >40% (e.g. UC Santa Cruz/El Camino)) or Open Admission"
    },
    "ecs": {
        "nat-int": "integer Number of National/International Activities where they are in a notable position",
        "reg": "integer Number of State/Regional Activities where they are in a notable position",
        "local": "integer Number of Local/School Activties/Clubs where they are in a notable position",
        "volunteering": "integer Number of Volunteering Activties they do",
        "ent": "integer Number of Entrepreneurial ventures (founded or co-founded)",
        "intern": "integer Number of internships",
        "add": "integer Number of classes taken outside of their high school curriculum",
        "res": "integer Number of research papers they have completed.",
        "sports": "integer Number of sports teams they are on, such as JV or Varsity",
        "work_exp": "integer Number of part-time jobs or significant ongoing family responsibilities",
        "leadership": "integer Total number of leadership positions held across all activities",
        "community_impact": "One integer number representing the depth and impact of community involvement where 0: No involvement, 1: Minimal involvement, 2: Moderate involvement, 3: Significant involvement, 4: Exceptional involvement with measurable impact",
        "ec_years": "One integer number representing the average number of years of involvement across all significant extracurricular activities"
    },
    "awards": {
        "int": "integer Number of International Awards (1st-10th place or equivalent)",
        "nat": "integer Number of National Awards (1st-10th place or equivalent)",
        "state": "integer Number of State/Regional Awards (1st-10th place or equivalent)",
        "local": "integer Number of Local/School Awards (1st-3rd place or equivalent)",
        "other": "integer Number of Honorable Mentions or Participation Awards (any level), do not count same award twice"
    }
}


json_string = json.dumps(JSON_SCHEMAS, indent=2)

def process_post(post_id):
    retries = 0
    while retries < 10:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Note:\nThe post is sourced from a Reddit forum. If it lacks sufficient information to make a determination, contains clearly false information, or is obviously a joke, output only {{\"skip\": true}}\n\nCRITICAL:\n\nYou MUST output a result for EVERY field even if not stated in the post.\nIf some piece of information is not stated in the post, make your best guess. But, if there is too much missing information, only output {{\"skip\": true}}\nDo not include any dialogue, only valid JSON.\n\nYou must only output valid JSON in this format:\n{json_string}\n\nHowever, if the post is a \"troll\" post, obviously fake, or contains too much missing information for the post to be accurately categorized/analyzed, only output {{\"skip\": true}}\n\nHere is the Reddit Post:\n{data[post_id]['link_flair_text']}\n{data[post_id]['selftext']}"}
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            message = completion.choices[0].message.content
            parse_and_append(message, "categorization/categorized.json")
            print(f"Processed post id: {post_id}")
            return
        except Exception as e:
            print(f"Error processing post id {post_id}: {e}")
            retries += 1
            if retries < 10:
                print(f"Retrying in 5 seconds... (Attempt {retries + 1}/10)")
                time.sleep(5)
            else:
                print(f"Failed to process post id {post_id} after 10 attempts.")

def worker(queue):
    while True:
        post_id = queue.get()
        if post_id is None:
            break
        process_post(post_id)
        queue.task_done()

def categorize():
    num_threads = 4
    queue = Queue()

    # Start worker threads
    threads = []
    for _ in range(num_threads):
        t = threading.Thread(target=worker, args=(queue,))
        t.start()
        threads.append(t)

    # Add post_ids to the queue
    for post_id in post_ids:
        queue.put(post_id)

    # Block until all tasks are done
    queue.join()

    # Stop workers
    for _ in range(num_threads):
        queue.put(None)
    for t in threads:
        t.join()

if __name__ == "__main__":
    categorize()