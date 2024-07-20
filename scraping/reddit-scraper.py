import praw
from prawcore.exceptions import PrawcoreException, ResponseException, RequestException
from dotenv import load_dotenv
import os
import json
from datetime import datetime
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=os.getenv("client_id"),
    client_secret=os.getenv("client_secret"),
    user_agent="nochances"
)

subreddit = reddit.subreddit("collegeresults")

# Output file path
output_file = "collegeresults_data.json"

# Date range
from_date = datetime.strptime("2022-01-01", "%Y-%m-%d")
to_date = datetime.now()

def load_existing_data():
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_data(data):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_last_post_date(existing_data):
    if existing_data:
        return max(datetime.utcfromtimestamp(post['time']) for post in existing_data)
    return from_date

def fetch_posts(existing_data, max_posts=2000):
    last_post_date = get_last_post_date(existing_data)
    after = None
    new_posts_count = 0
    total_posts = len(existing_data)
    
    logging.info(f"Starting to fetch posts after {last_post_date}")

    while new_posts_count < max_posts:
        try:
            new_posts = list(subreddit.new(limit=100, params={'after': after}))
            if not new_posts:
                logging.info("No more posts to fetch")
                break
            
            for post in new_posts:
                created = datetime.utcfromtimestamp(post.created_utc)
                if created <= last_post_date:
                    logging.info(f"Reached posts from or before {last_post_date}")
                    return existing_data
                if created > to_date:
                    continue

                selftext = post.selftext.strip()
                link_flair_text = post.link_flair_text

                # Filter out [removed], [deleted], and null link_flair_text
                if selftext in ["[removed]", "[deleted]"] or link_flair_text is None:
                    continue

                new_post_data = {
                    "id": post.id,
                    "time": post.created_utc,
                    "link_flair_text": link_flair_text,
                    "selftext": selftext
                }

                # Check if post already exists
                if not any(existing_post['id'] == post.id for existing_post in existing_data):
                    existing_data.append(new_post_data)
                    new_posts_count += 1
                    total_posts += 1

                if new_posts_count % 100 == 0:
                    logging.info(f"Processed {new_posts_count} new posts")
                    save_data(existing_data)  # Save data every 100 new posts

                if new_posts_count >= max_posts:
                    break
            
            after = new_posts[-1].fullname
            time.sleep(2)  # Add a small delay to avoid hitting rate limits
        
        except ResponseException as e:
            if e.response.status_code == 429:
                logging.warning("Rate limit exceeded. Waiting for 5 minutes before retrying.")
                time.sleep(300)  # Wait for 5 minutes
            else:
                logging.error(f"ResponseException occurred: {e}")
                time.sleep(60)
        
        except (RequestException, PrawcoreException) as e:
            logging.error(f"Exception occurred: {e}")
            time.sleep(60)
        
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            time.sleep(60)
    
    return existing_data

if __name__ == "__main__":
    logging.info(f"From date {from_date.strftime('%Y-%m-%d')} to date {to_date.strftime('%Y-%m-%d')}")
    
    existing_data = load_existing_data()
    logging.info(f"Loaded {len(existing_data)} existing posts")
    
    updated_data = fetch_posts(existing_data)
    
    save_data(updated_data)
    
    logging.info(f"Total posts after update: {len(updated_data)}")
    if updated_data:
        earliest_date = min(datetime.utcfromtimestamp(post['time']) for post in updated_data)
        latest_date = max(datetime.utcfromtimestamp(post['time']) for post in updated_data)
        logging.info(f"Earliest post date: {earliest_date}")
        logging.info(f"Latest post date: {latest_date}")
    else:
        logging.warning("No posts were scraped.")