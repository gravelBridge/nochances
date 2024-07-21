from openai import OpenAI
from dotenv import load_dotenv
import json
import random

load_dotenv()
client = OpenAI()

with open('scraping/combined_collegeresults_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
post_ids = list(data.keys())

def categorize(out_file, prompt):
    output = open(out_file, 'w')
    
    result = {}

    for post_id in post_ids:
        message = vectorize(post_id, prompt)
        result.update(post_id, message)
        print("Processed post id: " + post_id)
    
    output.write(json.dumps(result))

def vectorize(post_id, prompt):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
                {"role": "system", "content": "You are an expert in categorization and objectivity."},
                {"role": "user", "content": f"{prompt} \n\n Here is the student stats\n {data[post_id]["link_flair_text"]} \n {data[post_id]["selftext"]}"}
            ],
            temperature=0.0,
            seed=42
        )
    message = []
    try:
        message = json.loads(completion.choices[0].message.content)
    except Exception as e:
        print(e, completion)

    return message

def sample_outputs(n, prompt, out_file):
    keys = random.sample(post_ids, n)
    for key in keys:
        out_file.write(data[key]['selftext'])
        out_file.write(str(vectorize(key, prompt)))

if __name__ == '__main__':
    prompt = open('categorization/prompt.txt', 'r')

    prompt = prompt.readlines()
    prompt = "\n".join(prompt)

    # categorize(out_file='categorization/output.json',
    #            prompt=prompt)

    with open('categorization/samples.txt', 'w') as sample_output_file:
        sample_outputs(20, prompt, sample_output_file)