import praw

reddit = praw.Reddit(
    client_id="SQ5K2phxyPX1IPzIATucSw",
    client_secret="w5la470xfb1h1yIlrSPw5nE0kYzi1w",
    user_agent="your_user_agent"
)

subreddit = reddit.subreddit("AskReddit")
for post in subreddit.hot(limit=10):
    print(post.title)