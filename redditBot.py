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
document_list = []
for doc in documents:
    document_list.append(doc.selftext)

# try:
#     NLP.topic_modeling(preprocessed_docs, num_topics=5)
# except Exception as e:
#     print(f"Error during topic modeling: {e}")

# try:
#     submission = subreddit.submit(title=title.text, selftext=body.text)
#     print(f"Submission created: {submission.url}")
# except Exception as e:
#     print(f"Error posting: {e}")