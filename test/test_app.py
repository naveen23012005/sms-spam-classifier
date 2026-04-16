import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from app import transform_text
import pickle

def test_transform_text():
    text = "Hello this is a test message!!!"
    result = transform_text(text)

    assert isinstance(result, str)
    assert len(result) > 0


def test_vectorizer_load():
    tfidf = pickle.load(open("vectorizer.pkl", "rb"))
    assert tfidf is not None


def test_model_load():
    model = pickle.load(open("model.pkl", "rb"))
    assert model is not None