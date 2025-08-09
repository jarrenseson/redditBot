import pytest
from nlp import NLP
import praw
import os
from dotenv import load_dotenv

@pytest.fixture
def document_list():
    load_dotenv()

    # Reddit Creds
    USERNAME = os.getenv("REDDIT_USERNAME")  # Changed to avoid Windows USERNAME conflict
    PASSWORD = os.getenv("PASSWORD")
    CLIENT_SECRET = os.getenv("CLIENT_SECRET")
    CLIENT_ID = os.getenv("CLIENT_ID")

    # Reddit Instance
    redditInstance = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        username=USERNAME,
        password=PASSWORD,
        user_agent="redditBot by /u/Turbulent_Estate7372"
    )
    subreddit = redditInstance.subreddit("writingprompts")
    documents = subreddit.hot(limit=100000)
    document_list = []
    for doc in documents:
        document_list.append(doc.title)
    return document_list

def test_process(document_list):
    processed_docs = NLP.process(document_list)
    assert len(processed_docs) == len(document_list)
    assert all(isinstance(doc, list) for doc in processed_docs)

def test_train_model(document_list):
    processed_docs = NLP.process(document_list)
    NLP.train_lda_model(processed_docs)

def test_calculate_ideal_topics_num(document_list):
    processed_docs = NLP.process(document_list)
    NLP.calculate_ideal_topics_num(processed_docs)

def test_get_topics():
    NLP.get_topics()