import re
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from textblob import TextBlob
import gensim
from gensim import corpora


class NLP:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    @staticmethod
    def preprocess_text(text):
        """
        Preprocess the input text by lowering case, removing links, html tags, puncuatation, and extra white space
        """

        # Lower case
        text = text.lower()

        # Remove links
        text = re.sub(r'http\s +','',text)

        # Remove HTML tags
        text = re.sub(r'<.*?>','',text)

        # Remove punctuation
        text = text.translate(str.maketrans('','',string.punctuation))

        # Remove extra white space
        text = ''.join(text.split())

        return text
    
    @staticmethod
    def tokenize_lemmatize_text(text):
        """
        Tokenize the input text into words/sentences and lemmatize them
        """
        # Tokenizing
        tokens = word_tokenize(text)
        sentences = sent_tokenize(text)

        # Stemming
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in tokens]

        # Lemmatizing
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    @staticmethod
    def named_entity_recognition(text):
        """
        Perform Named Entity Recognition (NER) on the input text
        """
        nlp = spacy.load("en_core_web_sm")

        # POS tagging
        doc = nlp(text)

        return doc

    @staticmethod
    def sentiment_analysis(text):
        """
        Perform sentiment analysis on the input text
        """
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity

        return sentiment
    
    @staticmethod
    def topic_modeling(documents):
        """
        Perform topic modeling on a list of texts
        """
        dictionary = corpora.Dictionary(doc.split() for doc in documents)
        corpus = [dictionary.doc2bow(doc.split()) for doc in documents]
