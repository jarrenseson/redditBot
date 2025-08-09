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
        return lemmatized_tokens

    @staticmethod
    def process(document_list):
        preprocessed_docs = [NLP.preprocess_text(doc) for doc in document_list]
        processed_docs = [NLP.tokenize_lemmatize_text(doc) for doc in preprocessed_docs]
        return processed_docs
    
    @staticmethod
    def calculate_ideal_topics_num(tokens):    
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]
        for num_topics in range(2, 21):
            lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=dictionary,
                                               num_topics=num_topics)
            coherence_model_lda = CoherenceModel(model=lda_model, texts=tokens, dictionary=dictionary, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            print(f'Number of Topics: {num_topics}, Coherence Score: {coherence_lda}')
            
      
    @staticmethod
    def train_lda_model(tokens):
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
                                               num_topics=18)
        lda_model.save("lda_model.model")
        dictionary.save("dictionary.dict")
    
    @staticmethod
    def get_topics():
        """
        Get the topics from the trained LDA model
        """
        topic_labels = {
            0: "Magic & Villainy",
            1: "Hero’s Personal Struggles & Relationships",
            2: "Fantasy Creatures & Battles",
            3: "Royalty & Political Intrigue",
            4: "Gods, Wishes & Humanity",
            5: "Ancient Magic & Worldbuilding",
            6: "Aliens & Planetary Exploration",
            7: "Heroic Transformations",
            8: "Human Nature & Moral Choices",
            9: "Weapons, Death & Alternate Realities",
            10: "Love, Life & Personal Desires",
            11: "Demons & Supernatural Conflicts",
            12: "Strange Events & Departures",
            13: "Hero-Villain Love-Hate Dynamics",
            14: "Gods, Demons & Fate",
            15: "Power, Souls & Rebirth",
            16: "Mortality & Life-Changing Realizations",
            17: "Saving Kingdoms & Great Quests"
            }

        lda_model = gensim.models.LdaMulticore.load(".\modelStuff\lda_model.model")
        if len(lda_model.print_topics()) == len(topic_labels):
            print("Topics and their labels:")
            for i, topic in enumerate(lda_model.print_topics()):
                print(f"Topic {i}: {topic_labels[i]}")
                print(f"Details: {topic}")
        
    @staticmethod
    def tag_post(post):
        """
        Tag a single post (string) with the most relevant topic
        """
        lda_model = gensim.models.LdaMulticore.load("./modelStuff/lda_model.model")
        dictionary = corpora.Dictionary.load("./modelStuff/dictionary.dict")
        
        # Process expects a list of documents, so wrap post in a list
        processed_post = NLP.process(post)  # processed_post is list of list of tokens

        # Get bow vector for the first (and only) document
        bow_vector = dictionary.doc2bow(processed_post[0])

        if not bow_vector:
            print("Warning: No valid tokens in post after preprocessing. Cannot infer topic.")
            return None

        # Get topic distribution for this post
        topic_distribution = lda_model.get_document_topics(bow_vector)
        print(f"Topic distribution: {topic_distribution}")

        # Find topic with highest probability
        best_topic = max(topic_distribution, key=lambda x: x[1])[0]
        print(f"Most relevant topic: {best_topic}")
        
        return best_topic