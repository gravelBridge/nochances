import json
import zstandard
import io

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_zst_data(file_path):
    zst_data = {}
    with open(file_path, 'rb') as fh:
        dctx = zstandard.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
        for line in text_stream:
            data = json.loads(line)
            zst_data[data['id']] = data
    return zst_data

def sort_post_ids(post_ids):
    return sorted(post_ids, key=lambda x: (len(x), x))

def find_praw_posts(combined_data, zst_data):
    sorted_ids = sort_post_ids(combined_data.keys())
    
    first_praw_post_id = None
    last_praw_post_id = None
    
    for post_id in sorted_ids:
        if post_id not in zst_data:
            if first_praw_post_id is None:
                first_praw_post_id = post_id
            last_praw_post_id = post_id
    
    return first_praw_post_id, last_praw_post_id

def main():
    # File paths
    combined_data_file = r"scraping/combined_collegeresults_data.json"
    zst_input_file = r"scraping/collegeresults_submissions.zst"

    # Load combined data
    combined_data = load_json_data(combined_data_file)
    print(f"Loaded {len(combined_data)} posts from combined data.")

    # Load ZST data
    zst_data = load_zst_data(zst_input_file)
    print(f"Loaded {len(zst_data)} posts from ZST data.")

    # Find the first and last PRAW posts
    first_praw_post_id, last_praw_post_id = find_praw_posts(combined_data, zst_data)

    if first_praw_post_id and last_praw_post_id:
        print(f"The first post ID from PRAW data is: {first_praw_post_id}")
        print(f"The last post ID from PRAW data is: {last_praw_post_id}")
    else:
        print("No PRAW posts found in the combined data.")

if __name__ == "__main__":
    main()