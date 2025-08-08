import pytest
from nlp import NLP

def test_process():
    documents = [
        "This is a test document.",
        "Another test document goes here."
    ]
    processed_docs = NLP.process(documents)
    assert len(processed_docs) == 2
    assert all(isinstance(doc, list) for doc in processed_docs)