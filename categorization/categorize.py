from openai import OpenAI
import json
from dotenv import load_dotenv
import threading
import string

load_dotenv()
client = OpenAI()

with open('scraping/combined_collegeresults_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

post_ids = list(data.keys())
split_data = [post_ids[i:i+62] for i in range(0, len(post_ids), 62)]

output = open('categorization/output.json', 'w')
prompt = open('categorization/prompt.txt', 'r')

prompt = prompt.readlines()
prompt = "\n".join(prompt)

result = []

def categorize():
    for subarray in split_data:
        for post_id in subarray:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in categorization and objectivity."},
                    {"role": "user", "content": prompt + "\n\n Here is the student stats\n" + data[post_id]["link_flair_text"] + "\n" + data[post_id]["selftext"]}
                ],
                temperature=0.0,
                seed=42
            )
            message = completion.choices[0].message.content
            try:
                message = json.loads(message)
            except Exception as e:
                print(e)
                continue
            result.append(message)
            print(message)
            print("Processed post id: " + post_id)

categorize()
output.write(json.dumps(result))