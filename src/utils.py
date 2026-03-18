import matplotlib.pyplot as plt


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
    plt.tight_layout()
    plt.savefig(save_file)


def plot_loss_curve(losses, save_file):
    plt.figure(figsize=(12, 8))

    number_of_points = 25
    total_epochs = len(losses)
    step = max(1, total_epochs // number_of_points)
    indices = list(range(1, total_epochs, step))

    sampled_epochs = [i + total_epochs - indices[-1] - 1 for i in indices]
    sampled_losses = [losses[i] for i in sampled_epochs]

    plt.plot(
        sampled_epochs,
        sampled_losses,
        "r--",
        linewidth=2,
        marker="o",
        markersize=7,
        alpha=0.8,
    )
    plt.ylim(0, max(sampled_losses) * 1.05)

    plt.xlabel("Epochs")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training Loss over Epochs")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_file)
