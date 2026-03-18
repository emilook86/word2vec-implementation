from word2vec_implementation import Word2Vec
import matplotlib.pyplot as plt
from pathlib import Path
from logger_config import setup_logging

log = setup_logging(__name__)

NUMBER_OF_EPOCHS = 20_000



def plot_scatterplot(embeddings, save_file):
    plt.figure(figsize=(20, 15))
    words = list(embeddings.keys())
    x_coords = [embeddings[word][0] for word in words]
    y_coords = [embeddings[word][1] for word in words]

    plt.scatter(x_coords, y_coords, edgecolors="purple", alpha=0.6)

    for idx, word in enumerate(words):
        plt.annotate(
            word,
            (x_coords[idx], y_coords[idx]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            alpha=0.9,
        )

    plt.title("Word2Vec Embeddings Visualisation for the Given .txt File")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimnesion 2")
    plt.grid(True, alpha=0.4)

    plt.savefig(save_file)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    text = Path("text_example.txt").read_text()
    w2v = Word2Vec(text, embedding_dimension=2, logger=log)
    w2v.train(epochs=NUMBER_OF_EPOCHS)

    embeddings = w2v.get_word_embeddings()

    plot_scatterplot(embeddings=embeddings, save_file="embeddings_visualisation.png")
