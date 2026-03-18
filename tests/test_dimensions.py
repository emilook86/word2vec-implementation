import pytest
from src.word2vec_implementation import Word2Vec


@pytest.fixture
def sample_text():
    text = (
        "This is the first sentence in which there are eleven words."
        "This is the second sentence."
    )
    return text


def test_dimensions(sample_text):
    w2v = Word2Vec(
        dataset=sample_text, context_window_size=2, embedding_dimension=3, batch_size=4
    )
    batch = w2v.inputs[:4]
    y_true = w2v.outputs[:4]
    y_pred = w2v.forward(batch)
    loss_value = w2v.loss(y_true, y_pred)
    grad_input, grad_output = w2v.backpropagation(y_true, 4)

    assert w2v.input_embeddings_matrix.shape == (12, 3)
    assert w2v.output_embeddings_matrix.shape == (12, 3)
    assert y_pred.shape == (4, 12)
    assert loss_value > 0
    assert grad_input.shape == grad_output.shape
