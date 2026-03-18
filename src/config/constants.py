from pathlib import Path

ROOT_DIR = Path.cwd()
LOGS_DIR = ROOT_DIR / "figures" / "logs"
PLOTS_DIR = ROOT_DIR / "figures" / "plots"
CONFIG_DIR = ROOT_DIR / "src" / "config"

NUMBER_OF_EPOCHS = 100_000
BATCH_SIZE = 8
SAVE_EVERY = 1_000
LEARNING_RATE = 0.0005
