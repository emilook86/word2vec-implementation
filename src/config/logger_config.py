import logging
from src.config.constants import LOGS_DIR


def setup_logging(name=__name__, log_file=LOGS_DIR / "example_logs.log"):

    log_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    log_handler = logging.FileHandler(log_file)
    log_handler.setFormatter(log_formatter)

    console_formatter = logging.Formatter("%(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)

    logging.basicConfig(level=logging.INFO, handlers=[log_handler, console_handler])

    return logging.getLogger(name)
