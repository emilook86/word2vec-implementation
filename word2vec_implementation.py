import numpy


class Word2Vec:
    def __init__(
        self,
        vocabulary_length: int = 10,
        context_window_size: int = 3,
        embedding_dimension: int = 5,
        learning_rate: float = 0.1,
    ):

        self.vocabulary_length = vocabulary_length
        self.context_window_size = context_window_size
        self.embedding_dimension = embedding_dimension
        self.learning_rate = learning_rate

    def forward(self):
        pass

    def backpropagation(self):
        pass

    def train_step(self):
        pass
