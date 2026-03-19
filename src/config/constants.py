from pathlib import Path

ROOT_DIR = Path.cwd()
LOGS_DIR = ROOT_DIR / "figures" / "logs"
PLOTS_DIR = ROOT_DIR / "figures" / "plots"
CONFIG_DIR = ROOT_DIR / "src" / "config"

CONTEXT_WINDOW_SIZE = 2
EMBEDDING_DIMENSION = 2
LEARNING_RATE = 0.005
SAVE_EVERY = 2_000
BATCH_SIZE = 16
NUMBER_OF_EPOCHS = 100_000
