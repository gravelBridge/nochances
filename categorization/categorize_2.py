from openai import OpenAI
import json
from dotenv import load_dotenv
import time
import threading
from queue import Queue
import concurrent.futures

load_dotenv()
client = OpenAI()

with open("scraping/combined_collegeresults_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

post_ids = list(data.keys())

def categorize(out_file, prompt, as_json=False):
    try:
        output_json = json.load(open(out_file,"r+"))
    except:
        output_json = {}

    for post_id in post_ids:
        message = vectorize(post_id, prompt, as_json=as_json)
        output_json.update({post_id: message})
        print("Processed post id: " + post_id)
    
        json.dump(output_json, open(out_file,"r+"), indent=4)

def vectorize(post_id, prompt, as_json=False):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
                {"role": "system", "content": "You are an expert in categorization and objectivity."},
                {"role": "user", "content": f"{prompt} \n\n Here is the student stats\n {data[post_id]["link_flair_text"]} \n {data[post_id]["selftext"]}"}
            ],
            temperature=0.0,
            seed=42,
            response_format={"type": "json_object"} if as_json else None
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
    prompt2 = open('categorization/prompt_2.txt', 'r')

    prompt2 = prompt2.readlines()
    prompt2 = "\n".join(prompt2)

    categorize('categorization/output_2.json', prompt2, True)
    # print(vectorize('18tzh7x', prompt2, True))

    # with open('categorization/samples.txt', 'w') as sample_output_file:
    #     sample_outputs(20, prompt, sample_output_file)