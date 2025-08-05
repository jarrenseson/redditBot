import os
from dotenv import load_dotenv
import praw

load_dotenv()

USERNAME = os.getenv("REDDIT_USERNAME")  # Changed to avoid Windows USERNAME conflict
PASSWORD = os.getenv("PASSWORD")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
CLIENT_ID = os.getenv("CLIENT_ID")

print(repr(USERNAME), repr(CLIENT_ID), repr(CLIENT_SECRET), repr(PASSWORD))

redditInstance = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    username=USERNAME,
    password=PASSWORD,
    user_agent="redditBot by /u/Turbulent_Estate7372"
)

subreddit = redditInstance.subreddit("testingground4bots")
try:
    submission = subreddit.submit(title="Test Submission", selftext="This is a test submission for the bot.")
    print(f"Submission created: {submission.url}")
except Exception as e:
    print(f"Error fetching submissions: {e}")

# # prints logged in user
# try:
#     print(redditInstance.user.me())
# except Exception as e:
#     print(f"Error fetching user: {e}")