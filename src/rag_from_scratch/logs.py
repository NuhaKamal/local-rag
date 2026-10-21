"""
logs.py: set up logging once for the whole project.

Every module only SENDS messages:

    import logging
    logger = logging.getLogger(__name__)
    logger.info("embedded %d chunks", n)

and this file decides WHERE they go and HOW they look. Modules never
configure logging themselves, which keeps them decoupled.

LEVELS (from most to least detail):
  DEBUG    fine detail: scores, token counts
  INFO     progress: "index saved", "embedded 50/230"
  WARNING  something odd, but the program carries on
  ERROR    something failed

TWO OUTPUTS:
  screen : shows messages at the chosen level (normally INFO, so it stays quiet)
  file   : always records everything (DEBUG), with timestamps, in .cache/rag.log

RUN IT ON ITS OWN:
  uv run python -m rag_from_scratch.logs
"""

import logging
import tempfile
from pathlib import Path

from rag_from_scratch.config import LOG_LEVEL, LOG_PATH

PACKAGE_LOGGER = "rag_from_scratch"   # every module logger is a child of this name


def setup_logging(level: str = LOG_LEVEL, log_file: Path | None = LOG_PATH) -> logging.Logger:
    """Send the package's log messages to the screen (at `level`) and to a file (everything)."""
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.setLevel(logging.DEBUG)   # the handlers below decide what is actually shown
    logger.propagate = False         # do not also pass messages to Python's root logger

    for old in list(logger.handlers):   # calling this twice must not print every line twice
        old.close()
        logger.removeHandler(old)

    screen = logging.StreamHandler()    # writes to stderr, so it never mixes with the answer
    screen.setLevel(level.upper())
    screen.setFormatter(logging.Formatter("%(levelname)-7s %(module)s: %(message)s"))
    logger.addHandler(screen)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        to_file = logging.FileHandler(log_file, encoding="utf-8")
        to_file.setLevel(logging.DEBUG)
        to_file.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(module)s: %(message)s")
        )
        logger.addHandler(to_file)

    return logger


def _demo() -> None:
    print("== logs.py demo ==")
    with tempfile.TemporaryDirectory() as folder:
        log_file = Path(folder) / "demo.log"
        logger = setup_logging(level="INFO", log_file=log_file)

        print("INPUT : four messages sent from a module, one per level\n")
        print("ON SCREEN (level INFO: the DEBUG line is hidden):")
        demo = logging.getLogger("rag_from_scratch.demo")
        demo.debug("top score 0.71 on page 45")
        demo.info("embedded 50/230 chunks")
        demo.warning("index was built with another model")
        demo.error("model file not found")

        print("\nIN THE FILE (everything, with timestamps):")
        for line in log_file.read_text(encoding="utf-8").splitlines():
            print(f"  {line}")

        for handler in list(logger.handlers):   # release the file so the folder can be deleted
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    _demo()