import json

with open('scraping/combined_collegeresults_data.json','r') as f:
    post_json = json.load(f)

i = 0
for post in post_json.values():
    if 'MIT ' in post['selftext'].upper() or 'Massachusetts Institute of Technology'.upper() in post['selftext'].upper():
        i += 1

print(i)