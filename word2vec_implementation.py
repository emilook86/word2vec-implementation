import numpy as np


class Word2Vec:
    def __init__(
        self,
        dataset: list[str],
        context_window_size: int = 2,
        embedding_dimension: int = 3,
        learning_rate: float = 0.1,
        random_seed: int = 42,
    ):
        np.random.seed(random_seed)

        self.dataset = self.process_vocabulary(dataset)
        self.context_window_size = context_window_size
        self.embedding_dimension = embedding_dimension
        self.learning_rate = learning_rate

        self.vocabulary, self.vocabulary_size = self.create_vocabulary()

        self.inputs, self.outputs = self.create_input_output()

        self.input_embeddings_matrix = self.create_embeddings()
        self.output_embeddings_matrix = self.create_embeddings()

        self.cache = None

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

        vocabulary = {word: idx for idx, word in enumerate(sorted(list(vocabulary)))}
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

    def create_input_output(self):
        inputs = []
        outputs = []

        for sentence in self.dataset:
            words = sentence.split()
            sentence_length = len(words)
            if sentence_length < 2 * self.context_window_size + 1:
                continue

            for idx in range(
                self.context_window_size, sentence_length - self.context_window_size
            ):
                input_vector = np.zeros(self.vocabulary_size)
                output_vector = np.zeros(self.vocabulary_size)

                output_vector[self.vocabulary[words[idx]]] += 1
                for offset in range(1, self.context_window_size + 1):
                    input_vector[self.vocabulary[words[idx - offset]]] += 1
                    input_vector[self.vocabulary[words[idx + offset]]] += 1

                inputs.append(input_vector)
                outputs.append(output_vector)

        inputs = np.array(inputs)
        outputs = np.array(outputs)
        return inputs, outputs

    def forward(self, input_vector):
        """Parse only one vector each time"""
        hidden_layer = np.array(input_vector) @ self.input_embeddings_matrix

        output_layer = hidden_layer @ self.output_embeddings_matrix.T
        softmax_output_score = np.exp(output_layer) / np.sum(np.exp(output_layer))

        self.cache = {
            "input_vector": input_vector,
            "hidden_layer": hidden_layer,
            "softmax_output_score": softmax_output_score,
        }

        return softmax_output_score

    def loss(self, y_true, y_pred):
        target_idx = np.argmax(y_true)

        target_prob = y_pred[target_idx]
        epsilon = 1e-15
        target_prob = np.clip(target_prob, epsilon, 1 - epsilon)

        loss_value = -np.log(target_prob)
        return loss_value

    def backpropagation(self, y_true):
        target_idx = np.argmax(y_true)

        softmax_output_score = self.cache["softmax_output_score"]
        grad_output_score = softmax_output_score.copy()
        grad_output_score[target_idx] -= 1

        hidden_layer = self.cache["hidden_layer"]
        grad_output_embeddings = np.outer(grad_output_score, hidden_layer)

        grad_hidden_layer = grad_output_score @ self.output_embeddings_matrix

        input_vector = self.cache["input_vector"]
        grad_input_embeddings = np.outer(input_vector, grad_hidden_layer)

        return grad_input_embeddings, grad_output_embeddings

    def update_weights(self, grad_input, grad_output):
        self.input_embeddings_matrix -= grad_input * self.learning_rate
        self.output_embeddings_matrix -= grad_output * self.learning_rate

    def train_step(self, input_vector, y_true):
        y_pred = self.forward(input_vector)
        loss_value = self.loss(y_true, y_pred)
        grad_input_embeddings, grad_output_embeddings = self.backpropagation(y_true)
        self.update_weights(grad_input_embeddings, grad_output_embeddings)
        return loss_value

    def train():
        pass


if __name__ == "__main__":
    dataset = [
        "Hello there! What a beautiful day outside",
        "This photo is really beautiful",
        "Is that a bird? Is that a plane? No, that's Superman!",
    ]
    w2v = Word2Vec(dataset)
    print(w2v.inputs)
    print(w2v.outputs)
    print(w2v.vocabulary)
    print(w2v.input_embeddings_matrix)
    print("AND NOW")
    print(w2v.outputs[0])
    print(w2v.forward(w2v.inputs[0]))
    print(w2v.outputs[0].shape)
    print(w2v.forward(w2v.inputs[0]).shape)
    print(w2v.inputs.shape)
    print(w2v.input_embeddings_matrix.shape)
    print(w2v.backpropagation(1))
    print(f"Initial loss: {w2v.train_step(w2v.inputs[0], w2v.outputs[0])}")
