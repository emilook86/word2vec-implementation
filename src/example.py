from pathlib import Path

from src.word2vec_implementation import Word2Vec
from src.utils import plot_scatterplot, plot_loss_curve
from src.config.logger_config import setup_logging
from src.config.constants import (
    PLOTS_DIR,
    CONFIG_DIR,
    CONTEXT_WINDOW_SIZE,
    EMBEDDING_DIMENSION,
    LEARNING_RATE,
    SAVE_EVERY,
    BATCH_SIZE,
    NUMBER_OF_EPOCHS,
)

log = setup_logging(__name__)


if __name__ == "__main__":
    text = Path(CONFIG_DIR / "text_example.txt").read_text()
    w2v = Word2Vec(
        dataset=text,
        context_window_size=CONTEXT_WINDOW_SIZE,
        embedding_dimension=EMBEDDING_DIMENSION,
        logger=log,
        learning_rate=LEARNING_RATE,
        save_every=SAVE_EVERY,
        batch_size=BATCH_SIZE,
    )

    losses = w2v.train(epochs=NUMBER_OF_EPOCHS)
    embeddings = w2v.get_word_embeddings()

    if w2v.embedding_dimension == 2:
        save_scatterplot_file = PLOTS_DIR / "embeddings_visualisation.png"
        plot_scatterplot(embeddings=embeddings, save_file=save_scatterplot_file)

    save_losscurve_file = PLOTS_DIR / "losses_curve.png"
    plot_loss_curve(losses=losses, save_file=save_losscurve_file)
