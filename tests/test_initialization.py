import pytest
from src.word2vec_implementation import Word2Vec


@pytest.fixture
def sample_text():
    text = (
        "This is the first sentence in which there are eleven words."
        "This is the second sentence."
    )
    return text


def test_vocabulary_size(sample_text):
    w2v = Word2Vec(dataset=sample_text)
    assert w2v.vocabulary_size == 12


def test_training_set_length(sample_text):
    w2v_cws2 = Word2Vec(dataset=sample_text, context_window_size=2)
    w2v_cws5 = Word2Vec(dataset=sample_text, context_window_size=5)
    assert w2v_cws2.training_set_length == 8
    assert w2v_cws5.training_set_length == 1
