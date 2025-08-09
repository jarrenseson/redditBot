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
from gensim.models.coherencemodel import CoherenceModel


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

        # Remove tags
        text = re.sub(r'wp|sp', ' ', text)

        return text
    
    @staticmethod
    def tokenize_lemmatize_text(text):
        """
        Tokenize the input text into words/sentences and lemmatize them
        """
        stop_words = set(stopwords.words('english'))
        stop_words.update({
            'wp', 'sp', 'prompt', 'story', 'write', 'writing', 'writeup', 'written', 'tale',
            'fiction', 'author', 'character', 'characters', 'plot', 'scene', 'chapter', 'narrative',
            'read', 'reads', 'writingprompt', 'prompted', 'requests', 'one', 'like', 'time', 'know',
            'said', 'would', 'could', 'even', 'get', 'back', 'people', 'year', 'day', 'make', 'see',
            'go', 'want', 'think', 'way', 'look', 'come', 'take', 'find', 'say', 'i', 'you', 'he',
            'she', 'they', 'we', 'will', 'shall', 'must', 'might', 'should', 'one', 'time', 'youre'
        })



        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)

        lemmatized_tokens = [
            lemmatizer.lemmatize(token)
            for token in tokens
            if token not in stop_words and token.isalpha()
        ]
        print(f"Lemmatized Tokens: {lemmatized_tokens}")
        return lemmatized_tokens
    
    @staticmethod
    def topic_modeling(tokens, num_topics=6):
        """
        Perform topic modeling on the input documents using LDA
        """
        # Create a dictionary for LDA
        dictionary = corpora.Dictionary(tokens)

        # Create a corpus
        corpus = [dictionary.doc2bow(token) for token in tokens]

        # Train LDA model
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=dictionary,
                                               num_topics=num_topics)
        # for idx, topic in lda_model.print_topics(-1):
        #     print(f"Topic {idx}: {topic}")

    @staticmethod
    def process(document_list):
        preprocessed_docs = [NLP.preprocess_text(doc) for doc in document_list]
        processed_docs = [NLP.tokenize_lemmatize_text(doc) for doc in preprocessed_docs]

        return processed_docs
    
    @staticmethod
    def calculate_ideal_topics_num(tokens):    
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        for num_topics in range(2, 16,2):
            lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=dictionary,
                                               num_topics=num_topics)
            coherence_model_lda = CoherenceModel(model=lda_model, texts=tokens, dictionary=dictionary, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            print(f"Number of Topics: {num_topics}, Coherence Score: {coherence_lda}")
        