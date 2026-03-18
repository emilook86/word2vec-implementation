import matplotlib.pyplot as plt
from pathlib import Path

from src.word2vec_implementation import Word2Vec
from src.config.logger_config import setup_logging
from src.config.constants import (
    PLOTS_DIR,
    NUMBER_OF_EPOCHS,
    BATCH_SIZE,
    SAVE_EVERY,
    CONFIG_DIR,
    LEARNING_RATE,
)

log = setup_logging(__name__)


def plot_scatterplot(embeddings, save_file):
    plt.figure(figsize=(16, 12))
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
    text = Path(CONFIG_DIR / "text_example.txt").read_text()
    w2v = Word2Vec(
        dataset=text,
        embedding_dimension=2,
        logger=log,
        learning_rate=LEARNING_RATE,
        save_every=SAVE_EVERY,
        batch_size=BATCH_SIZE,
    )

    w2v.train(epochs=NUMBER_OF_EPOCHS)

    embeddings = w2v.get_word_embeddings()

    save_file = PLOTS_DIR / "embeddings_visualisation.png"
    plot_scatterplot(embeddings=embeddings, save_file=save_file)
