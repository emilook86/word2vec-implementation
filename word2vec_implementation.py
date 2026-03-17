import numpy as np


class Word2Vec:
    def __init__(
        self,
        dataset: list[str],
        context_window_size: int = 3,
        embedding_dimension: int = 5,
        learning_rate: float = 0.1,
    ):

        self.dataset = dataset
        self.context_window_size = context_window_size
        self.embedding_dimension = embedding_dimension
        self.learning_rate = learning_rate

        self.vocabulary, self.vocabulary_size = self.create_vocabulary()
        self.embeddings_matrix = self.create_embeddings()

    def create_vocabulary(self):
        vocabulary = set()
        for sentence in self.dataset:
            line = sentence.split(" ")
            for word in line:
                word = "".join(char for char in word if char.isalpha())
                word = word.lower()
                vocabulary.add(word)
        vocabulary = list(vocabulary)
        vocabulary_size = len(vocabulary)
        return vocabulary, vocabulary_size

    def create_embeddings(self):
        max_value = np.sqrt(1 / self.vocabulary_size)
        embeddings_matrix = []
        for _ in range(self.vocabulary_size):
            embeddings_matrix.append(
                np.random.uniform(
                    low=-max_value, high=max_value, size=self.embedding_dimension
                )
            )
        embeddings_matrix = np.array(embeddings_matrix)
        return embeddings_matrix

    def forward(self):
        pass

    def backpropagation(self):
        pass

    def train_step(self):
        pass


vocabulary = set()


print(vocabulary)
