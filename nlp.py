# Libraries
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import spacy
from textblob import TextBlob
import gensim
from gensim import corpora
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')


class NLP:
    def __init__(self, documents):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.coherence_score = None
        self.documents = documents

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

        return text
    
    @staticmethod
    def tokenize_lemmatize_text(text):
        """
        Tokenize the input text into words/sentences and lemmatize them
        """
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

        tokens = word_tokenize(text)
        lemmatized_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token not in stop_words and token.isalpha()
        ]
        return lemmatized_tokens
    
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
        Determine optimal amount of topics + perform topic modeling on a list of texts
        """
        topic_coherence_scores = dict()
        for topic_num in range(2, 11, 2):
            # Create a dictionary and corpus for the documents
            dictionary = corpora.Dictionary(doc.split() for doc in documents)
            corpus = [dictionary.doc2bow(doc.split()) for doc in documents]

            lda = gensim.models.LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=topic_num)
            
