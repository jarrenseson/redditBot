import os
from dotenv import load_dotenv
import praw
from google import genai
from nlp.nlp import NLP


load_dotenv()

# Reddit Creds
USERNAME = os.getenv("REDDIT_USERNAME")  # Changed to avoid Windows USERNAME conflict
PASSWORD = os.getenv("PASSWORD")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
CLIENT_ID = os.getenv("CLIENT_ID")

# Gemini Creds
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Reddit Instance
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    username=USERNAME,
    password=PASSWORD,
    user_agent="redditBot by /u/Turbulent_Estate7372"
)

# Gemini Instance
gemini = genai.Client()

# Setting subreddit to bot test subreddit
subreddit = reddit.subreddit("shortstories")
# documents = [reddit.submission(url="https://www.reddit.com/r/shortstories/comments/k67uhy/ms_i_pretended_to_be_a_missing_girl/")]
documents = subreddit.new(limit=1)
submission = None
document_list = []
for doc in documents:
    submission = doc
    print(f"url: {doc.url}")
    document_list.append(doc.selftext)

topic = NLP.tag_post(document_list)

body = gemini.models.generate_content(
    model="gemini-2.5-flash", contents=f"Act as a seasoned writer. Generate a response to the following post: {document_list[0]}, here is the topic extraced using LDA: {topic}. Make sure to stay on topic"
)

try:
    comment = submission.reply(body.text)
    print(f"Comment created: {comment.permalink}")
except Exception as e:
    print(f"Error posting comments: {e}")

# try:
#     NLP.topic_modeling(preprocessed_docs, num_topics=5)
# except Exception as e:
#     print(f"Error during topic modeling: {e}")

# try:
#     submission = subreddit.submit(title=title.text, selftext=body.text)
#     print(f"Submission created: {submission.url}")
# except Exception as e:
#     print(f"Error posting: {e}")