import os
from dotenv import load_dotenv
import praw
from google import genai
from nlp import NLP


load_dotenv()

# Reddit Creds
USERNAME = os.getenv("REDDIT_USERNAME")  # Changed to avoid Windows USERNAME conflict
PASSWORD = os.getenv("PASSWORD")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
CLIENT_ID = os.getenv("CLIENT_ID")

# Gemini Creds
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Reddit Instance
redditInstance = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    username=USERNAME,
    password=PASSWORD,
    user_agent="redditBot by /u/Turbulent_Estate7372"
)

# # Gemini Instance
# gemini = genai.Client()

# title = gemini.models.generate_content(
#     model="gemini-2.5-flash", contents="Give me a single eyecatching title for a Reddit post about random fun facts about anything. Give just a title, no other nonsense in your response"
# )

# body = gemini.models.generate_content(
#     model="gemini-2.5-flash", contents=f"Write an interesting fun fact that only a very small percentage of people know about. Make it suitable for a Reddit post. The fact should be interesting and engaging, and it should be something that would spark curiosity or discussion among readers. Keep it concise and to the point."
# )

# print(f"Gemini title: {title.text}")
# print(f"Gemini Body: {body.text}")
# Setting subreddit to bot test subreddit
subreddit = redditInstance.subreddit("shortstories")
documents = subreddit.hot(limit=10)

preprocessed_docs = [NLP.preprocess_text(doc.selftext) for doc in documents]
print(f"Preprocessed Documents: {preprocessed_docs}")
processed_docs = [NLP.tokenize_lemmatize_text(doc) for doc in preprocessed_docs]
print(f"Processed Documents: {processed_docs}")

# try:
#     submission = subreddit.submit(title=title.text, selftext=body.text)
#     print(f"Submission created: {submission.url}")
# except Exception as e:
#     print(f"Error posting: {e}")