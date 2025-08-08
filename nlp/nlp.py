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

        # Remove escape characters
        text = re.sub(r'[\r\n\t]+', ' ', text)

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
    def topic_modeling(tokens, num_topics=5):
        """
        Perform topic modeling on the input documents using LDA
        """
        # Create a dictionary for LDA
        dictionary = corpora.Dictionary(tokens)

        # Create a corpus
        unicode_tokens = []
        for i, token in enumerate(tokens):
            for char in token:
                unicode_tokens[i] = unicode_tokens[i] + (r'\\u{:04X}'.format(ord(char)))
                print(unicode_tokens[i])
        corpus = [dictionary.doc2bow(unicode_token) for unicode_token in unicode_tokens]

        # Train LDA model
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=dictionary,
                                               num_topics=num_topics)
        print(lda_model.print_topics())

    @staticmethod
    def process(document_list):
        preprocessed_docs = [NLP.preprocess_text(doc) for doc in document_list]
        processed_docs = [NLP.tokenize_lemmatize_text(doc) for doc in preprocessed_docs]

        return processed_docs