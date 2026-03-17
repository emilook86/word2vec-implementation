import numpy as np


class Word2Vec:
    def __init__(
        self,
        dataset: list[str],
        context_window_size: int = 3,
        embedding_dimension: int = 5,
        learning_rate: float = 0.1,
    ):

        self.dataset = self.process_vocabulary(dataset)
        self.context_window_size = context_window_size
        self.embedding_dimension = embedding_dimension
        self.learning_rate = learning_rate

        self.vocabulary, self.vocabulary_size = self.create_vocabulary()
        self.embeddings_matrix = self.create_embeddings()

    @staticmethod
    def process_vocabulary(vocab):
        processed_vocab = []
        for sentence in vocab:
            words = sentence.split()
            processed_words = []
            for word in words:
                clean_word = "".join(char for char in word if char.isalpha())
                if clean_word:
                    processed_words.append(clean_word.lower())
            processed_vocab.append(" ".join(processed_words))
        return processed_vocab

    def create_vocabulary(self):
        vocabulary = set()
        for sentence in self.dataset:
            vocabulary.update(sentence.split())
        vocabulary = {idx: word for idx, word in enumerate(list(vocabulary))}
        vocabulary_size = len(vocabulary)
        return vocabulary, vocabulary_size

    def create_embeddings(self):
        max_value = np.sqrt(1 / self.vocabulary_size)
        embeddings_matrix = np.random.uniform(
            low=-max_value,
            high=max_value,
            size=(self.vocabulary_size, self.embedding_dimension),
        )
        return embeddings_matrix

    def forward(self):
        pass

    def backpropagation(self):
        pass

    def train_step(self):
        pass
