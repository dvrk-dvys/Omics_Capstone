import logging
import time
from contextlib import contextmanager
from rich.console import Console
from rich.logging import RichHandler

# Shared console — used by both RichHandler and all Progress/track() calls
# so that log messages and progress bars never fight over the terminal.
console = Console()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = RichHandler(console=console, show_path=False, markup=False)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def log_duration(logger: logging.Logger, label: str):
    """Context manager that logs elapsed time for a block."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        if elapsed >= 60:
            logger.info(f"⏱️  {label} completed in {elapsed / 60:.1f} min")
        else:
            logger.info(f"⏱️  {label} completed in {elapsed:.1f}s")
