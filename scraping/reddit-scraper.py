import praw
from dotenv import load_dotenv
import os

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("client_id"),
    client_secret=os.getenv("client_secret"),
    user_agent="nochances"
)

subreddit = reddit.subreddit("AskReddit")
for post in subreddit.hot(limit=10):
    print(post.title)