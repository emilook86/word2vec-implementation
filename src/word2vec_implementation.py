import numpy as np


class Word2Vec:
    def __init__(
        self,
        dataset: str,
        context_window_size: int = 2,
        embedding_dimension: int = 3,
        random_seed: int = 42,
        logger=None,
        learning_rate: float = 0.01,
        save_every: int = 500,
        batch_size: int = 16,
    ):
        np.random.seed(random_seed)

        self.dataset = self.process_vocabulary(dataset)
        self.context_window_size = context_window_size
        self.embedding_dimension = embedding_dimension
        self.logger = logger
        self.learning_rate = learning_rate
        self.save_every = save_every
        self.batch_size = batch_size

        self.vocabulary, self.vocabulary_size = self.create_vocabulary()

        self.inputs, self.outputs = self.create_input_output()
        self.training_set_length = len(self.inputs)

        self.input_embeddings_matrix = self.create_embeddings()
        self.output_embeddings_matrix = self.create_embeddings()

        self.cache = None

    @staticmethod
    def process_vocabulary(vocab):
        if "." not in vocab:
            raise Exception("The text must have at least one dot")

        vocab = vocab.split(".")
        vocab = vocab[:-1]

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
                input_vector /= np.sum(input_vector)

                inputs.append(input_vector)
                outputs.append(output_vector)

        inputs = np.array(inputs)
        outputs = np.array(outputs)
        return inputs, outputs

    def forward(self, input_batch):
        """Parse only batch each time"""
        hidden_layer = input_batch @ self.input_embeddings_matrix

        output_layer = hidden_layer @ self.output_embeddings_matrix.T
        output_layer -= np.max(output_layer, axis=1, keepdims=True)

        exp_scores = np.exp(output_layer)
        softmax_output_score = exp_scores / (
            1e-15 + np.sum(exp_scores, axis=1, keepdims=True)
        )

        self.cache = {
            "input_batch": input_batch,
            "hidden_layer": hidden_layer,
            "softmax_output_score": softmax_output_score,
        }

        return softmax_output_score

    def loss(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss_value = -np.sum(y_true * np.log(y_pred), axis=1)
        mean_loss = np.mean(loss_value)
        return mean_loss

    def backpropagation(self, y_true, batch_size):
        softmax_output_score = self.cache["softmax_output_score"]
        grad_output_score = (softmax_output_score.copy() - y_true) / batch_size

        hidden_layer = self.cache["hidden_layer"]
        grad_output_embeddings = grad_output_score.T @ hidden_layer

        grad_hidden_layer = grad_output_score @ self.output_embeddings_matrix

        input_batch = self.cache["input_batch"]
        grad_input_embeddings = input_batch.T @ grad_hidden_layer

        return grad_input_embeddings, grad_output_embeddings

    def update_weights(self, grad_input, grad_output):
        self.input_embeddings_matrix -= grad_input * self.learning_rate
        self.output_embeddings_matrix -= grad_output * self.learning_rate

    def train_step(self, input_batch, y_true_batch):
        y_pred_batch = self.forward(input_batch)
        loss_value = self.loss(y_true_batch, y_pred_batch)
        grad_input_embeddings, grad_output_embeddings = self.backpropagation(
            y_true_batch, batch_size=input_batch.shape[0]
        )
        self.update_weights(grad_input_embeddings, grad_output_embeddings)
        return loss_value

    def train(self, epochs=100):
        n_examples = len(self.inputs)
        losses = []

        for epoch_idx in range(epochs):
            total_loss = 0
            indices = np.random.permutation(n_examples)

            for start in range(0, n_examples, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                input_batch = self.inputs[batch_idx]
                output_batch = self.outputs[batch_idx]
                loss = self.train_step(input_batch, output_batch)
                total_loss += loss * len(batch_idx)

            if self.logger and epoch_idx % self.save_every == 0:
                self.logger.info(
                    f"Total Cross-Entropy Loss after epoch {epoch_idx} equals to {total_loss}."
                )
            losses.append(total_loss)

        return losses

    def get_word_embeddings(self):
        word_embeddings = dict()
        for word in self.vocabulary:
            word_idx = self.vocabulary[word]
            word_embeddings[word] = self.input_embeddings_matrix[word_idx]
        return word_embeddings
