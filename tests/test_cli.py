import pytest

from rag_from_scratch.cli import build_parser, main, screen_level
from rag_from_scratch.config import CHUNK_SIZE, OVERLAP


@pytest.fixture(autouse=True)
def no_real_logging(monkeypatch):
    """main() sets up logging; keep tests from writing to the real .cache/rag.log."""
    monkeypatch.setattr("rag_from_scratch.cli.setup_logging", lambda **kwargs: None)


def test_ask_collects_words_and_options():
    args = build_parser().parse_args(["ask", "what", "is", "a", "score", "-k", "5", "--no-llm"])
    assert args.question == ["what", "is", "a", "score"]
    assert args.top_k == 5
    assert args.no_llm is True


def test_ingest_defaults():
    args = build_parser().parse_args(["ingest"])
    assert args.pdfs == []
    assert args.chunk_size == CHUNK_SIZE
    assert args.overlap == OVERLAP
    assert args.verbose == 0


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


@pytest.mark.parametrize("flags, level", [([], "WARNING"), (["-v"], "INFO"), (["-vv"], "DEBUG")])
def test_verbose_flag_sets_the_screen_level(flags, level):
    args = build_parser().parse_args(["ask", "hello", *flags])
    assert screen_level(args.verbose) == level


def test_ask_without_an_index_fails_politely(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("rag_from_scratch.cli.INDEX_PATH", tmp_path / "missing.json")
    assert main(["ask", "hello"]) == 1
    assert "ingest" in capsys.readouterr().err