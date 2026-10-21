import logging

import pytest

from rag_from_scratch.logs import PACKAGE_LOGGER, setup_logging


@pytest.fixture(autouse=True)
def clean_logger():
    """Close the log file handlers after each test (Windows cannot delete open files)."""
    yield
    logger = logging.getLogger(PACKAGE_LOGGER)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def test_file_gets_everything_even_when_the_screen_is_quiet(tmp_path, capsys):
    log_file = tmp_path / "rag.log"
    setup_logging(level="WARNING", log_file=log_file)

    module_logger = logging.getLogger("rag_from_scratch.demo")  # same name style as __name__
    module_logger.debug("detail for the file")
    module_logger.warning("heads up")

    text = log_file.read_text(encoding="utf-8")
    assert "detail for the file" in text   # the file keeps DEBUG
    assert "heads up" in text
    screen = capsys.readouterr().err
    assert "heads up" in screen            # the screen shows WARNING
    assert "detail for the file" not in screen


def test_calling_setup_twice_does_not_duplicate_lines(tmp_path):
    log_file = tmp_path / "rag.log"
    setup_logging(level="INFO", log_file=log_file)
    setup_logging(level="INFO", log_file=log_file)

    logging.getLogger("rag_from_scratch.demo").info("only once")
    assert log_file.read_text(encoding="utf-8").count("only once") == 1


def test_file_lines_have_timestamps(tmp_path):
    log_file = tmp_path / "rag.log"
    setup_logging(level="INFO", log_file=log_file)
    logging.getLogger("rag_from_scratch.demo").info("hello")
    line = log_file.read_text(encoding="utf-8").splitlines()[0]
    assert line[:4].isdigit()              # starts with the year of the timestamp
    assert "INFO" in line and "hello" in line