# Word2Vec NumPy Implementation

This project implements a **Word2Vec training loop from scratch** using only **NumPy**. It uses the **CBOW (Continuous Bag of Words)** approach, predicting a target word from its surrounding context, and covers text preprocessing, input-output pair creation, embedding initialization, forward pass, loss computation, backpropagation, and gradient updates.

## Requirements

- [uv](https://docs.astral.sh/uv/) package manager
- Linux, macOS, or Windows

## Quick Start

```bash
# Installation
git clone https://github.com/emilook86/word2vec-implementation.git
cd word2vec-implementation
uv sync

# Activate environment
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\Activate.ps1

# If you want to run the training loop
uv run src/example.py
```


## Project Structure

src/
- word2vec_implementation.py - main class: vocabulary creation, input/output generation, embeddings, forward/backward pass, training loop.
- example.py - run this to train on a text (default: `src/config/text_example.txt` containing the song "Fly me to the Earth" by Wallace Collection), generate embeddings and loss plots.
- utils.py - preprocessing raw text, plotting embeddings and loss curves.
- config/
  - constants.py - hyperparameters, file paths.
  - logger_config.py - logging setup.
  - text_example.txt - sample text dataset.

tests/
- test_dimensions.py - checks shapes and gradient computations.
- test_initialization.py - checks vocabulary and training set.

figures/
- logs/ - training logs.
- plots/
  - embeddings_visualisation.png - 2D plot with words by embedding values.
  - losses_curve.png - training curve plot.

## Usage Notes

- Modify hyperparameters in `src/config/constants.py`.  
- Training is executed via `src/example.py`.
- The text is separated to training sentences by one or multiple ".", "!", "?", ";", ":" delimiters or end-of-line character (implemented with regex in `src/utils.py`).  
- After training, embeddings can be visualized in `figures/plots/embeddings_visualisation.png` and training losses curve in `figures/plots/losses_curve.png`.  

## Implementation Highlights

- CBOW: input is the **context words**, output is the **center word**.  
- Supports **context window-based input-output pairs** with one-hot encoding output vectors.  
- Embeddings initialized randomly; learned through gradient descent.  
- Forward pass uses **softmax**; loss is **cross-entropy**.  
- Backpropagation computes gradients for both input and output embeddings.  
- Mini-batch training with configurable batch size, learning rate, and save frequency.  
- Final embeddings returned as a dictionary mapping words to vectors.
