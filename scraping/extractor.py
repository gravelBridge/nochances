import praw
from prawcore.exceptions import PrawcoreException
from dotenv import load_dotenv
import os
import json
from datetime import datetime
import time
import zstandard
import logging

# Set up logging
log = logging.getLogger("bot")
log.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
log_str_handler = logging.StreamHandler()
log_str_handler.setFormatter(log_formatter)
log.addHandler(log_str_handler)

# Load environment variables
load_dotenv()

# Reddit API setup
reddit = praw.Reddit(
    client_id=os.getenv("client_id"),
    client_secret=os.getenv("client_secret"),
    user_agent="nochances"
)

subreddit = reddit.subreddit("collegeresults")

# File paths
zst_input_file = r"scraping/collegeresults_submissions.zst"
output_file = r"scraping/combined_collegeresults_data.json"

# Date range
from_date = datetime.strptime("2000-01-01", "%Y-%m-%d")
to_date = datetime.strptime("2030-12-31", "%Y-%m-%d")

def fetch_posts(max_posts=2000):
    post_data = {}
    after = None
    while len(post_data) < max_posts:
        try:
            new_posts = list(subreddit.new(limit=100, params={'after': after}))
            if not new_posts:
                break  # No more posts to fetch
            
            for post in new_posts:
                post_data[post.id] = {
                    'link_flair_text': post.link_flair_text,
                    'selftext': post.selftext
                }
                
                if len(post_data) % 100 == 0:
                    log.info(f"Processed {len(post_data)} posts from Reddit API")
                
                if len(post_data) >= max_posts:
                    break
            
            after = new_posts[-1].fullname
            time.sleep(2)  # Add a small delay to avoid hitting rate limits
        
        except PrawcoreException as e:
            log.error(f"An error occurred: {e}")
            time.sleep(60)  # Wait for a minute before retrying
    
    return post_data

def read_and_decode(reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0):
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(f"Unable to decode frame after reading {bytes_read:,} bytes")
        log.info(f"Decoding error with {bytes_read:,} bytes, reading another chunk")
        return read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)

def read_lines_zst(file_name):
    with open(file_name, 'rb') as file_handle:
        buffer = ''
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(file_handle)
        while True:
            chunk = read_and_decode(reader, 2**27, (2**29) * 2)
            if not chunk:
                break
            lines = (buffer + chunk).split("\n")
            for line in lines[:-1]:
                yield line.strip(), file_handle.tell()
            buffer = lines[-1]
        reader.close()

def process_zst_file(input_file, from_date, to_date):
    log.info(f"Processing ZST file: {input_file}")
    
    file_size = os.stat(input_file).st_size
    created = None
    matched_lines = 0
    filtered_lines = 0
    total_lines = 0
    results = {}

    for line, file_bytes_processed in read_lines_zst(input_file):
        total_lines += 1
        if total_lines % 100000 == 0:
            log.info(f"{created.strftime('%Y-%m-%d %H:%M:%S')} : {total_lines:,} : {matched_lines:,} : {filtered_lines:,} : {file_bytes_processed:,}:{(file_bytes_processed / file_size) * 100:.0f}%")

        try:
            obj = json.loads(line)
            created = datetime.utcfromtimestamp(int(obj['created_utc']))

            if created < from_date or created > to_date:
                continue

            selftext = obj.get("selftext", "").strip()
            link_flair_text = obj.get("link_flair_text")

            # Filter out [removed], [deleted], and null link_flair_text
            if selftext in ["[removed]", "[deleted]"] or link_flair_text is None:
                filtered_lines += 1
                continue

            matched_lines += 1
            results[obj["id"]] = {
                "link_flair_text": link_flair_text,
                "selftext": selftext
            }

        except (KeyError, json.JSONDecodeError) as err:
            log.warning(f"Error processing line: {err}")
            log.warning(line)

    log.info(f"ZST processing complete : {total_lines:,} : {matched_lines:,} : {filtered_lines:,}")
    return results

def combine_data(zst_data, api_data):
    combined_data = zst_data.copy()
    for post_id, post_info in api_data.items():
        if post_id not in combined_data:
            combined_data[post_id] = post_info
    return combined_data

def main():
    log.info(f"Starting data collection and processing")
    
    # Process ZST file
    zst_data = process_zst_file(zst_input_file, from_date, to_date)
    log.info(f"ZST data processed: {len(zst_data)} entries")

    # Fetch data from Reddit API
    api_data = fetch_posts()
    log.info(f"API data fetched: {len(api_data)} entries")

    # Combine data
    combined_data = combine_data(zst_data, api_data)
    log.info(f"Combined data: {len(combined_data)} entries")

    # Write combined data to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=2)

    log.info(f"Combined data written to {output_file}")

if __name__ == "__main__":
    main()